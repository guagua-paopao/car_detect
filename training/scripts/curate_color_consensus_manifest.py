#!/usr/bin/env python3
"""Deduplicate and color-by-vehicle cap hard-scene color consensus rows."""

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


def rank(row: dict, prefer_night: bool = False) -> tuple:
    small = row.get("vehicle_size") == "small"
    night = truthy(row.get("night"))
    hard_count = sum(truthy(row.get(key)) for key in ("occluded", "truncated")) + int(small)
    confidence = float(row.get("color_teacher_confidence") or 0)
    tie = hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest()
    return (-int(night and prefer_night), -int(small), -hard_count, -confidence, tie)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-per-color-vehicle", type=int, default=40)
    parser.add_argument("--max-per-color", type=int, default=120)
    parser.add_argument("--max-per-source-image", type=int, default=1)
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    parser.add_argument("--prefer-night", action="store_true")
    parser.add_argument("--minimum-night-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite curated color evidence")
    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    candidates = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("color_teacher_consensus") == "accepted"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("formal_train_eligible"))
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["color"], row["official_vehicle_class"])].append(row)
    selected = []
    source_images = Counter()
    selected_color_counts = Counter()
    selected_hashes: dict[str, list[str]] = defaultdict(list)
    rejected = Counter()
    for color, vehicle in sorted(grouped):
        cell_selected = 0
        for row in sorted(
            grouped[(color, vehicle)], key=lambda item: rank(item, args.prefer_night)
        ):
            if cell_selected >= args.max_per_color_vehicle:
                rejected["color_vehicle_cell_cap"] += 1
                continue
            if selected_color_counts[color] >= args.max_per_color:
                rejected["color_cap"] += 1
                continue
            source_image = row.get("source_image_id") or row.get("source_frame_id")
            if source_images[source_image] >= args.max_per_source_image:
                rejected["source_image_cap"] += 1
                continue
            dhash = row.get("dhash64", "")
            if len(dhash) != 16:
                rejected["invalid_dhash"] += 1
                continue
            if any(hamming(dhash, previous) <= args.max_dhash_distance for previous in selected_hashes[color]):
                rejected["perceptual_near_duplicate"] += 1
                continue
            row["review_status"] = "approved_agent_color_consensus_after_visual_review"
            row["review_method"] = (
                row.get("review_method", "")
                + "+dhash_dedup+source_image_cap+color_vehicle_cap"
            )
            selected.append(row)
            source_images[source_image] += 1
            selected_color_counts[color] += 1
            selected_hashes[color].append(dhash)
            cell_selected += 1
    selected.sort(key=lambda row: (
        row["color"], row["official_vehicle_class"], rank(row, args.prefer_night)
    ))
    if not selected:
        raise RuntimeError("no rows survived color consensus curation")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    cell_counts = Counter((row["color"], row["official_vehicle_class"]) for row in selected)
    small = sum(row.get("vehicle_size") == "small" for row in selected)
    occluded = sum(truthy(row.get("occluded")) for row in selected)
    truncated = sum(truthy(row.get("truncated")) for row in selected)
    night = sum(truthy(row.get("night")) for row in selected)
    quotas = {
        "small_target_fraction": small / len(selected),
        "small_target_fraction_at_least_0_20": small / len(selected) >= 0.20,
        "occluded_fraction": occluded / len(selected),
        "occluded_fraction_at_least_0_15": occluded / len(selected) >= 0.15,
        "truncated_fraction": truncated / len(selected),
        "truncated_fraction_at_least_0_15": truncated / len(selected) >= 0.15,
        "night_fraction": night / len(selected),
        "night_fraction_at_least_required_when_enabled": (
            not args.prefer_night or night / len(selected) >= args.minimum_night_fraction
        ),
    }
    low_support = sorted(color for color, count in selected_color_counts.items() if count < 20)
    quota_gates = [value for key, value in quotas.items() if "at_least" in key]
    status = "pass" if (
        len(selected) >= 500
        and len(selected_color_counts) >= 8
        and min(selected_color_counts.values()) >= 8
        and all(quota_gates)
    ) else "fail"
    report = {
        "schema_version": "color-consensus-curation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "teacher_accepted_candidates": len(candidates),
        "source_color_counts": dict(sorted(Counter(row["color"] for row in candidates).items())),
        "source_color_vehicle_counts": {
            f"{color}|{vehicle}": count
            for (color, vehicle), count in sorted(Counter(
                (row["color"], row["official_vehicle_class"]) for row in candidates
            ).items())
        },
        "max_per_color_vehicle": args.max_per_color_vehicle,
        "max_per_color": args.max_per_color,
        "max_per_source_image": args.max_per_source_image,
        "max_dhash_distance": args.max_dhash_distance,
        "selected_rows": len(selected),
        "selected_unique_source_images": len(source_images),
        "selected_color_counts": dict(sorted(selected_color_counts.items())),
        "selected_color_vehicle_counts": {
            f"{color}|{vehicle}": count for (color, vehicle), count in sorted(cell_counts.items())
        },
        "low_support_colors_below_20": low_support,
        "selected_small_target_rows": small,
        "selected_occluded_rows": occluded,
        "selected_truncated_rows": truncated,
        "selected_night_rows": night,
        "prefer_night": args.prefer_night,
        "minimum_night_fraction": args.minimum_night_fraction,
        "quotas": quotas,
        "rejection_counts": dict(sorted(rejected.items())),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "train_only": True,
            "teacher_accepted_only": True,
            "color_by_vehicle_shortcut_capped": True,
            "perceptual_near_duplicates_excluded": True,
            "one_crop_per_source_image": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "use only as a low-weight hard-scene color supplement; merge with passenger-vehicle and long-tail color sources before training"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
