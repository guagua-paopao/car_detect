#!/usr/bin/env python3
"""Deduplicate and cap fail-closed body consensus rows for a train-only supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CAPS = {
    "bus": 600,
    "heavy_truck": 500,
    "light_truck": 200,
    "sedan": 300,
    "suv": 200,
    "mpv": 200,
    "van": 200,
    "pickup": 200,
    "other": 200,
}


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
    confidence = float(row.get("teacher_consensus_confidence") or 0)
    tie = hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest()
    return (-int(night and prefer_night), -int(small), -hard_count, -confidence, tie)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-source-class-count", type=int, default=20)
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    parser.add_argument("--max-per-source-image", type=int, default=1)
    parser.add_argument("--prefer-night", action="store_true")
    parser.add_argument("--minimum-night-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite curated consensus evidence")
    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    candidates = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("teacher_consensus") == "accepted"
        and truthy(row.get("body_type_supervised"))
        and truthy(row.get("formal_train_eligible"))
    ]
    source_counts = Counter(row["body_type"] for row in candidates)
    eligible_classes = {
        label for label, count in source_counts.items()
        if count >= args.minimum_source_class_count
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row["body_type"] in eligible_classes:
            grouped[row["body_type"]].append(row)
    selected = []
    selected_source_images = Counter()
    selected_hashes: dict[str, list[str]] = defaultdict(list)
    rejected = Counter()
    by_class_rejected = Counter()
    for label in sorted(grouped):
        cap = DEFAULT_CAPS.get(label, 200)
        for row in sorted(grouped[label], key=lambda item: rank(item, args.prefer_night)):
            if sum(item["body_type"] == label for item in selected) >= cap:
                rejected["class_cap"] += 1
                by_class_rejected[f"{label}:class_cap"] += 1
                continue
            source_image = row.get("source_image_id") or row.get("source_frame_id")
            if selected_source_images[source_image] >= args.max_per_source_image:
                rejected["source_image_cap"] += 1
                by_class_rejected[f"{label}:source_image_cap"] += 1
                continue
            dhash = row.get("dhash64", "")
            if len(dhash) != 16:
                rejected["invalid_dhash"] += 1
                by_class_rejected[f"{label}:invalid_dhash"] += 1
                continue
            if any(hamming(dhash, previous) <= args.max_dhash_distance for previous in selected_hashes[label]):
                rejected["perceptual_near_duplicate"] += 1
                by_class_rejected[f"{label}:perceptual_near_duplicate"] += 1
                continue
            row["review_status"] = "approved_agent_consensus_after_visual_review"
            row["review_method"] = (
                row.get("review_method", "")
                + "+dhash_dedup+source_image_cap+body_class_cap"
            )
            selected.append(row)
            selected_source_images[source_image] += 1
            selected_hashes[label].append(dhash)
    selected.sort(key=lambda row: (row["body_type"], rank(row, args.prefer_night)))
    if not selected:
        raise RuntimeError("no rows survived body consensus curation")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    selected_counts = Counter(row["body_type"] for row in selected)
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
    status = "pass" if (
        len(selected) >= 1000
        and len(selected_counts) >= 5
        and all(
            value for key, value in quotas.items()
            if key.endswith("at_least_0_20")
            or key.endswith("at_least_0_15")
            or key == "night_fraction_at_least_required_when_enabled"
        )
    ) else "fail"
    report = {
        "schema_version": "body-consensus-curation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "accepted_teacher_candidates": len(candidates),
        "source_body_type_counts": dict(sorted(source_counts.items())),
        "minimum_source_class_count": args.minimum_source_class_count,
        "eligible_classes": sorted(eligible_classes),
        "excluded_low_count_classes": sorted(set(source_counts) - eligible_classes),
        "class_caps": DEFAULT_CAPS,
        "max_per_source_image": args.max_per_source_image,
        "max_dhash_distance": args.max_dhash_distance,
        "selected_rows": len(selected),
        "selected_unique_source_images": len(selected_source_images),
        "selected_body_type_counts": dict(sorted(selected_counts.items())),
        "selected_small_target_rows": small,
        "selected_occluded_rows": occluded,
        "selected_truncated_rows": truncated,
        "selected_night_rows": night,
        "prefer_night": args.prefer_night,
        "minimum_night_fraction": args.minimum_night_fraction,
        "quotas": quotas,
        "rejection_counts": dict(sorted(rejected.items())),
        "rejection_counts_by_class": dict(sorted(by_class_rejected.items())),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "train_only": True,
            "teacher_accepted_only": True,
            "low_count_classes_fail_closed": True,
            "perceptual_near_duplicates_excluded": True,
            "one_crop_per_source_image": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "use only as a bounded low-weight hard-scene supplement after manifest merge checks; never replace the balanced base corpus"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
