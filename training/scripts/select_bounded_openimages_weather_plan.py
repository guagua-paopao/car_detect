#!/usr/bin/env python3
"""Select a deterministic, quota-aware and byte-bounded weather image plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_targets(values: list[str], name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key, separator, count = value.partition("=")
        if not separator or not key or not count.isdigit():
            raise ValueError(f"invalid {name} target {value!r}; expected label=nonnegative_integer")
        result[key] = int(count)
    return result


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--min-images", type=int, default=1000)
    parser.add_argument("--weather-target", action="append", default=[])
    parser.add_argument("--vehicle-target", action="append", default=[])
    parser.add_argument("--minimum-night-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite bounded-plan evidence")
    if args.max_images <= 0 or args.max_bytes <= 0:
        raise ValueError("max-images and max-bytes must be positive")

    weather_targets = parse_targets(args.weather_target, "weather")
    vehicle_targets = parse_targets(args.vehicle_target, "vehicle")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("status") != "pass":
        raise RuntimeError("source weather plan did not pass")
    if source_report.get("output_plan_sha256") != sha256(args.source_plan):
        raise RuntimeError("source weather plan hash mismatch")
    if source_report.get("policy", {}).get("train_split_only") is not True:
        raise RuntimeError("source plan is not train-only")
    if source_report.get("policy", {}).get("frozen_video_used") is not False:
        raise RuntimeError("source plan policy does not prove frozen-video isolation")

    with args.source_plan.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError("source plan is empty")

    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, 2):
        image_id = row.get("image_id", "")
        if not image_id:
            raise ValueError(f"source row {line_number}: empty image_id")
        by_image[image_id].append(row)

    records: dict[str, dict] = {}
    weather_available: Counter[str] = Counter()
    vehicle_available: Counter[str] = Counter()
    for image_id, image_rows in by_image.items():
        sizes = {int(row.get("original_size_bytes") or 0) for row in image_rows}
        md5s = {row.get("original_md5_base64", "") for row in image_rows}
        licenses = {row.get("license_url", "") for row in image_rows}
        if len(sizes) != 1 or min(sizes) <= 0 or len(md5s) != 1 or "" in md5s:
            raise ValueError(f"inconsistent size or MD5 metadata for {image_id}")
        if len(licenses) != 1 or not all("creativecommons.org/licenses/by/2.0" in x for x in licenses):
            raise ValueError(f"invalid image license for {image_id}")
        weathers = {
            value
            for row in image_rows
            for value in row.get("weather_labels", "").split(";")
            if value
        }
        vehicles = {row.get("vehicle_class", "") for row in image_rows} - {""}
        hard = {
            "small": any(truthy(row.get("small_target_proxy")) for row in image_rows),
            "occluded": any(truthy(row.get("occluded")) for row in image_rows),
            "truncated": any(truthy(row.get("truncated")) for row in image_rows),
        }
        records[image_id] = {
            "image_id": image_id,
            "rows": image_rows,
            "bytes": next(iter(sizes)),
            "weather": weathers,
            "vehicles": vehicles,
            "hard": hard,
        }
        for label in weathers:
            weather_available[label] += 1
        for label in vehicles:
            vehicle_available[label] += 1

    unknown_weather = sorted(set(weather_targets) - set(weather_available))
    unknown_vehicle = sorted(set(vehicle_targets) - set(vehicle_available))
    if unknown_weather or unknown_vehicle:
        raise ValueError(f"targets reference missing strata: weather={unknown_weather}, vehicle={unknown_vehicle}")

    def priority(record: dict) -> tuple:
        hard_count = sum(record["hard"].values())
        # Prefer genuinely hard and multi-condition images, then lower bytes.
        return (
            -hard_count,
            -len(record["weather"]),
            -len(record["vehicles"]),
            record["bytes"],
            record["image_id"],
        )

    selected: set[str] = set()
    selected_bytes = 0

    def can_add(record: dict) -> bool:
        return (
            record["image_id"] not in selected
            and len(selected) < args.max_images
            and selected_bytes + record["bytes"] <= args.max_bytes
        )

    def add(record: dict) -> bool:
        nonlocal selected_bytes
        if not can_add(record):
            return False
        selected.add(record["image_id"])
        selected_bytes += record["bytes"]
        return True

    weather_selected: Counter[str] = Counter()
    vehicle_selected: Counter[str] = Counter()

    def refresh_counts(record: dict) -> None:
        for label in record["weather"]:
            weather_selected[label] += 1
        for label in record["vehicles"]:
            vehicle_selected[label] += 1

    # Scarce strata are filled first. One image can satisfy several quotas.
    strata = [
        ("weather", label, target, weather_available[label])
        for label, target in weather_targets.items()
    ] + [
        ("vehicle", label, target, vehicle_available[label])
        for label, target in vehicle_targets.items()
    ]
    strata.sort(key=lambda item: (item[3] / max(1, item[2]), item[0], item[1]))
    ordered = sorted(records.values(), key=priority)
    for kind, label, target, _available in strata:
        counter = weather_selected if kind == "weather" else vehicle_selected
        record_key = "weather" if kind == "weather" else "vehicles"
        candidates = [
            record for record in ordered
            if label in record[record_key] and record["image_id"] not in selected
        ]
        for record in candidates:
            if counter[label] >= target:
                break
            if add(record):
                refresh_counts(record)

    for record in ordered:
        if len(selected) >= args.max_images:
            break
        if add(record):
            refresh_counts(record)

    output_rows = [row for image_id in sorted(selected) for row in by_image[image_id]]
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    hard_image_counts = Counter()
    for image_id in selected:
        for condition, present in records[image_id]["hard"].items():
            hard_image_counts[condition] += present
        hard_image_counts["union"] += any(records[image_id]["hard"].values())
    quota_failures = {}
    for label, target in weather_targets.items():
        if weather_selected[label] < target:
            quota_failures[f"weather:{label}"] = {"target": target, "actual": weather_selected[label]}
    for label, target in vehicle_targets.items():
        if vehicle_selected[label] < target:
            quota_failures[f"vehicle:{label}"] = {"target": target, "actual": vehicle_selected[label]}
    night_fraction = weather_selected["night"] / len(selected) if selected else 0.0
    gates = {
        "minimum_images": len(selected) >= args.min_images,
        "maximum_images": len(selected) <= args.max_images,
        "maximum_bytes": selected_bytes <= args.max_bytes,
        "minimum_night_fraction": night_fraction >= args.minimum_night_fraction,
        "all_requested_strata_targets": not quota_failures,
    }
    status = "pass_bounded_plan" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-bounded-weather-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_plan": str(args.source_plan.resolve()),
        "source_plan_sha256": sha256(args.source_plan),
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256(args.source_report),
        "source_unique_images": len(records),
        "source_rows": len(rows),
        "available_weather_image_counts": dict(sorted(weather_available.items())),
        "available_vehicle_image_counts": dict(sorted(vehicle_available.items())),
        "selection_limits": {
            "max_images": args.max_images,
            "max_bytes": args.max_bytes,
            "min_images": args.min_images,
            "minimum_night_fraction": args.minimum_night_fraction,
        },
        "requested_weather_image_targets": weather_targets,
        "requested_vehicle_image_targets": vehicle_targets,
        "selected_unique_images": len(selected),
        "selected_rows": len(output_rows),
        "selected_bytes": selected_bytes,
        "selected_weather_image_counts": dict(sorted(weather_selected.items())),
        "selected_vehicle_image_counts": dict(sorted(vehicle_selected.items())),
        "selected_hard_image_counts": dict(sorted(hard_image_counts.items())),
        "selected_night_fraction": night_fraction,
        "quota_failures": quota_failures,
        "gates": gates,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "deterministic": True,
            "all_boxes_for_selected_images_retained": True,
            "per_image_cc_by_2_license_rechecked": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "download only if this bounded plan passes and its quotas are reviewed",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "images": len(selected),
        "rows": len(output_rows),
        "bytes": selected_bytes,
        "weather": dict(sorted(weather_selected.items())),
        "vehicles": dict(sorted(vehicle_selected.items())),
        "hard": dict(sorted(hard_image_counts.items())),
        "night_fraction": night_fraction,
        "quota_failures": quota_failures,
    }, ensure_ascii=False))
    return 0 if status == "pass_bounded_plan" else 2


if __name__ == "__main__":
    raise SystemExit(main())
