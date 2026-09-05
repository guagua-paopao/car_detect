#!/usr/bin/env python3
"""Combine locked hard-scene colors with weather teacher consensus, fail closed."""

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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def rank(row: dict[str, str]) -> tuple:
    night = truthy(row.get("night"))
    small = row.get("vehicle_size") == "small"
    hard = int(small) + sum(truthy(row.get(key)) for key in ("occluded", "truncated"))
    confidence = float(
        row.get("color_teacher_confidence")
        or row.get("pseudo_label_confidence")
        or row.get("review_score")
        or 0
    )
    tie = hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest()
    return (-int(night), -int(small), -hard, -confidence, tie)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--existing-report", type=Path, required=True)
    parser.add_argument("--weather-manifest", type=Path, required=True)
    parser.add_argument("--weather-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-per-color-vehicle", type=int, default=80)
    parser.add_argument("--max-per-color", type=int, default=220)
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    parser.add_argument("--minimum-night-fraction", type=float, default=0.30)
    parser.add_argument("--maximum-neutral-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite combined color curation evidence")

    existing_report = json.loads(args.existing_report.read_text(encoding="utf-8"))
    if existing_report.get("status") != "pass":
        raise RuntimeError("existing color curation report did not pass")
    if existing_report.get("output_manifest_sha256") != sha256(args.existing_manifest):
        raise RuntimeError("existing color curation hash mismatch")
    weather_report = json.loads(args.weather_report.read_text(encoding="utf-8"))
    if weather_report.get("output_manifest_sha256") != sha256(args.weather_manifest):
        raise RuntimeError("weather color teacher manifest hash mismatch")
    policy = weather_report.get("policy", {})
    if not (
        policy.get("foreground_or_official_labels_retained_only_not_replaced") is True
        and policy.get("rejected_rows_changed_to_unknown_unsupervised") is True
        and policy.get("test_split_not_used") is True
        and policy.get("frozen_video_not_used") is True
    ):
        raise RuntimeError("weather color teacher policy is not strict or isolated")

    existing_fields, existing_rows = load_csv(args.existing_manifest)
    weather_fields, weather_rows = load_csv(args.weather_manifest)
    existing_candidates = [
        {**row, "combined_source_pool": "existing_hard"}
        for row in existing_rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("formal_train_eligible"))
        and row.get("color") not in {"", "unknown"}
    ]
    weather_candidates = [
        {**row, "combined_source_pool": "weather"}
        for row in weather_rows
        if row.get("split") == "train"
        and row.get("color_teacher_consensus") == "accepted"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("formal_train_eligible"))
        and row.get("color") not in {"", "unknown"}
    ]
    if len(weather_candidates) != int(weather_report.get("accepted_rows", -1)):
        raise RuntimeError("weather accepted-row count differs from color teacher report")
    if not weather_candidates:
        raise RuntimeError("weather color teacher produced no accepted rows")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in existing_candidates + weather_candidates:
        grouped[(row["color"], row.get("official_vehicle_class", "unknown"))].append(row)
    selected = []
    selected_hashes: dict[str, list[str]] = defaultdict(list)
    selected_source_images = set()
    selected_paths = set()
    selected_color_counts = Counter()
    rejected = Counter()
    for color, vehicle in sorted(grouped):
        cell_selected = 0
        for row in sorted(grouped[(color, vehicle)], key=rank):
            if cell_selected >= args.max_per_color_vehicle:
                rejected["color_vehicle_cell_cap"] += 1
                continue
            if selected_color_counts[color] >= args.max_per_color:
                rejected["color_cap"] += 1
                continue
            path = row.get("image_path", "")
            if not path or path in selected_paths:
                rejected["duplicate_or_empty_image_path"] += 1
                continue
            source_image = row.get("source_image_id") or row.get("source_frame_id") or path
            if source_image in selected_source_images:
                rejected["source_image_cap"] += 1
                continue
            dhash = row.get("dhash64", "")
            if len(dhash) != 16:
                rejected["invalid_dhash"] += 1
                continue
            if any(hamming(dhash, previous) <= args.max_dhash_distance for previous in selected_hashes[color]):
                rejected["perceptual_near_duplicate"] += 1
                continue
            row["review_status"] = "approved_agent_combined_color_consensus"
            row["formal_train_eligible"] = "true"
            row["sample_weight"] = "0.500" if row["combined_source_pool"] == "weather" else "0.350"
            row["review_method"] = (
                row.get("review_method", "")
                + "+combined_weather_existing_dhash_source_color_vehicle_curation"
            )
            selected.append(row)
            selected_paths.add(path)
            selected_source_images.add(source_image)
            selected_hashes[color].append(dhash)
            selected_color_counts[color] += 1
            cell_selected += 1
    selected.sort(key=lambda row: (row["color"], row.get("official_vehicle_class", ""), rank(row)))
    if not selected:
        raise RuntimeError("no rows survived combined color curation")

    fields = list(existing_fields)
    for field in [*weather_fields, "combined_source_pool", "sample_weight"]:
        if field not in fields:
            fields.append(field)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    source_counts = Counter(row["combined_source_pool"] for row in selected)
    cell_counts = Counter(
        (row["color"], row.get("official_vehicle_class", "unknown")) for row in selected
    )
    night = sum(truthy(row.get("night")) for row in selected)
    small = sum(row.get("vehicle_size") == "small" for row in selected)
    hard_union = sum(
        row.get("vehicle_size") == "small"
        or truthy(row.get("occluded"))
        or truthy(row.get("truncated"))
        for row in selected
    )
    neutral = selected_color_counts["white"] + selected_color_counts["silver_gray"]
    minimum_color_count = min(selected_color_counts.values()) if selected_color_counts else 0
    quotas = {
        "night_fraction": night / len(selected),
        "night_fraction_at_least_required": night / len(selected) >= args.minimum_night_fraction,
        "small_fraction": small / len(selected),
        "small_fraction_at_least_0_20": small / len(selected) >= 0.20,
        "hard_union_fraction": hard_union / len(selected),
        "hard_union_fraction_at_least_0_15": hard_union / len(selected) >= 0.15,
        "white_plus_silver_gray_fraction": neutral / len(selected),
        "neutral_fraction_at_most_required": neutral / len(selected) <= args.maximum_neutral_fraction,
    }
    status = "pass" if (
        len(selected) >= 800
        and len(selected_color_counts) >= 8
        and minimum_color_count >= 20
        and all(value for key, value in quotas.items() if "at_least" in key or "at_most" in key)
    ) else "fail"
    report = {
        "schema_version": "color-weather-combined-curation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "existing_manifest": str(args.existing_manifest.resolve()),
        "existing_manifest_sha256": sha256(args.existing_manifest),
        "existing_report": str(args.existing_report.resolve()),
        "existing_report_sha256": sha256(args.existing_report),
        "weather_manifest": str(args.weather_manifest.resolve()),
        "weather_manifest_sha256": sha256(args.weather_manifest),
        "weather_report": str(args.weather_report.resolve()),
        "weather_report_sha256": sha256(args.weather_report),
        "weather_teacher_report_status": weather_report.get("status"),
        "weather_teacher_status_note": "a fail caused only by standalone sample-count gates does not invalidate individually accepted exact-consensus rows; combined gates remain mandatory",
        "input_rows": {
            "existing_eligible": len(existing_candidates),
            "weather_teacher_accepted": len(weather_candidates),
        },
        "selected_rows": len(selected),
        "selected_unique_source_images": len(selected_source_images),
        "selected_source_pool_counts": dict(sorted(source_counts.items())),
        "selected_color_counts": dict(sorted(selected_color_counts.items())),
        "selected_color_vehicle_counts": {
            f"{color}|{vehicle}": count for (color, vehicle), count in sorted(cell_counts.items())
        },
        "minimum_selected_color_count": minimum_color_count,
        "selected_night_rows": night,
        "selected_small_rows": small,
        "selected_hard_union_rows": hard_union,
        "quotas": quotas,
        "max_per_color_vehicle": args.max_per_color_vehicle,
        "max_per_color": args.max_per_color,
        "max_dhash_distance": args.max_dhash_distance,
        "rejection_counts": dict(sorted(rejected.items())),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "existing_curated_rows_hash_locked": True,
            "weather_rows_require_foreground_and_exact_multiteacher_multiview_consensus": True,
            "weather_teacher_standalone_sample_gate_not_treated_as_label_quality": True,
            "combined_sample_scene_long_tail_and_neutral_color_gates_mandatory": True,
            "one_crop_per_source_image": True,
            "perceptual_near_duplicates_excluded_within_color": True,
            "color_by_official_vehicle_shortcut_capped": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only a pass-status combined report may enter the next training-manifest merge",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "rows": len(selected),
        "sources": dict(sorted(source_counts.items())),
        "colors": dict(sorted(selected_color_counts.items())),
        "quotas": quotas,
        "rejections": dict(sorted(rejected.items())),
    }, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
