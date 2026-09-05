#!/usr/bin/env python3
"""Select a deterministic byte-bounded machine-Night vehicle image plan."""

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


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def parse_targets(values: list[str]) -> dict[str, int]:
    targets = {}
    for value in values:
        label, separator, count = value.partition("=")
        if not separator or not label or not count.isdigit():
            raise ValueError(f"invalid vehicle target {value!r}; expected label=count")
        targets[label] = int(count)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=6000)
    parser.add_argument("--min-images", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--vehicle-target", action="append", default=[])
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite bounded machine-night plan evidence")
    targets = parse_targets(args.vehicle_target)

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("status") != "pass_source_plan_pending_photometric_audit":
        raise RuntimeError("machine-night source plan did not pass")
    if source_report.get("output_plan_sha256") != sha256(args.source_plan):
        raise RuntimeError("machine-night source plan hash mismatch")
    policy = source_report.get("policy", {})
    if policy.get("train_split_only") is not True or policy.get("frozen_video_used") is not False:
        raise RuntimeError("source plan does not prove train-only frozen-video isolation")

    with args.source_plan.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    if not by_image:
        raise RuntimeError("machine-night source plan is empty")

    records = []
    available = Counter()
    for image_id, image_rows in by_image.items():
        sizes = {int(row.get("original_size_bytes") or 0) for row in image_rows}
        md5s = {row.get("original_md5_base64", "") for row in image_rows}
        licenses = {row.get("license_url", "") for row in image_rows}
        confidences = {float(row["night_machine_confidence"]) for row in image_rows}
        if len(sizes) != 1 or min(sizes) <= 0 or len(md5s) != 1 or "" in md5s:
            raise ValueError(f"invalid size or MD5 metadata for {image_id}")
        if len(licenses) != 1 or not all("creativecommons.org/licenses/by/2.0" in x for x in licenses):
            raise ValueError(f"invalid per-image license for {image_id}")
        if len(confidences) != 1:
            raise ValueError(f"inconsistent machine Night confidence for {image_id}")
        vehicles = {row["vehicle_class"] for row in image_rows}
        for vehicle in vehicles:
            available[vehicle] += 1
        hard = sum(any(truthy(row.get(key)) for row in image_rows) for key in (
            "small_target_proxy", "occluded", "truncated"
        ))
        records.append({
            "image_id": image_id,
            "rows": image_rows,
            "bytes": next(iter(sizes)),
            "confidence": next(iter(confidences)),
            "vehicles": vehicles,
            "hard": hard,
        })

    missing = sorted(set(targets) - set(available))
    if missing:
        raise ValueError(f"vehicle targets reference missing strata: {missing}")
    records.sort(key=lambda record: (
        -int(bool(record["vehicles"] & {"bus", "truck"})),
        -record["hard"],
        -record["confidence"],
        record["bytes"],
        record["image_id"],
    ))
    selected: set[str] = set()
    selected_bytes = 0
    selected_vehicle = Counter()

    def add(record: dict) -> bool:
        nonlocal selected_bytes
        if record["image_id"] in selected or len(selected) >= args.max_images:
            return False
        if selected_bytes + record["bytes"] > args.max_bytes:
            return False
        selected.add(record["image_id"])
        selected_bytes += record["bytes"]
        for vehicle in record["vehicles"]:
            selected_vehicle[vehicle] += 1
        return True

    for vehicle, target in sorted(targets.items(), key=lambda item: available[item[0]]):
        for record in records:
            if selected_vehicle[vehicle] >= target:
                break
            if vehicle in record["vehicles"]:
                add(record)
    for record in records:
        if len(selected) >= args.max_images:
            break
        add(record)

    output_rows = [row for image_id in sorted(selected) for row in by_image[image_id]]
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    quota_failures = {
        vehicle: {"target": target, "actual": selected_vehicle[vehicle]}
        for vehicle, target in targets.items()
        if selected_vehicle[vehicle] < target
    }
    hard_counts = Counter()
    confidence_counts = Counter()
    record_by_id = {record["image_id"]: record for record in records}
    for image_id in selected:
        record = record_by_id[image_id]
        confidence_counts[f"{record['confidence']:.2f}"] += 1
        for key in ("small_target_proxy", "occluded", "truncated"):
            hard_counts[key] += any(truthy(row.get(key)) for row in record["rows"])
        hard_counts["union"] += record["hard"] > 0
    gates = {
        "minimum_images": len(selected) >= args.min_images,
        "maximum_images": len(selected) <= args.max_images,
        "maximum_bytes": selected_bytes <= args.max_bytes,
        "vehicle_targets": not quota_failures,
        "all_candidates_pending_pixel_photometric_audit": True,
    }
    status = "pass_bounded_plan_pending_photometric_audit" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-bounded-machine-night-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_plan": str(args.source_plan.resolve()),
        "source_plan_sha256": sha256(args.source_plan),
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256(args.source_report),
        "source_unique_images": len(by_image),
        "source_rows": len(rows),
        "available_vehicle_image_counts": dict(sorted(available.items())),
        "selection_limits": {
            "max_images": args.max_images,
            "min_images": args.min_images,
            "max_bytes": args.max_bytes,
        },
        "requested_vehicle_image_targets": targets,
        "selected_unique_images": len(selected),
        "selected_rows": len(output_rows),
        "selected_bytes": selected_bytes,
        "selected_vehicle_image_counts": dict(sorted(selected_vehicle.items())),
        "selected_hard_image_counts": dict(sorted(hard_counts.items())),
        "selected_machine_confidence_counts": dict(sorted(confidence_counts.items())),
        "quota_failures": quota_failures,
        "gates": gates,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "deterministic": True,
            "all_boxes_for_selected_images_retained": True,
            "machine_night_remains_candidate_metadata": True,
            "pixel_photometric_audit_required": True,
            "per_image_cc_by_2_license_rechecked": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "download only this bounded pool; reject pixels that fail photometric or teacher consistency",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "images": len(selected),
        "rows": len(output_rows),
        "bytes": selected_bytes,
        "vehicles": dict(sorted(selected_vehicle.items())),
        "hard": dict(sorted(hard_counts.items())),
        "quota_failures": quota_failures,
    }, ensure_ascii=False))
    return 0 if status == "pass_bounded_plan_pending_photometric_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
