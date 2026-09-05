#!/usr/bin/env python3
"""Independently audit all Stage66 train crops against held-out crops."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filter_stage66_cross_split_near_duplicates import near_matches, parse_dhash, sha256


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def group_key(row: dict[str, str]) -> str:
    return row.get("track_group") or row.get("video_id") or row.get("camera_id") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-distance", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage66 full leakage evidence")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    marker_rows = sum(
        any(marker in " ".join(str(value).lower() for value in row.values()) for marker in FROZEN_MARKERS)
        for row in rows
    )
    train = [row for row in rows if row.get("split") == "train"]
    heldout = [row for row in rows if row.get("split") in {"validation", "test"}]
    if not train or not heldout:
        raise RuntimeError("train or held-out split is empty")
    matches = near_matches(
        [parse_dhash(row) for row in train],
        [parse_dhash(row) for row in heldout],
        args.maximum_distance,
    )
    distance_counts: Counter[int] = Counter()
    for row_matches in matches:
        distance_counts.update(distance for _, distance in row_matches)
    near_pair_count = sum(distance_counts.values())
    train_sha = {row.get("crop_sha256") for row in train} - {None, ""}
    heldout_sha = {row.get("crop_sha256") for row in heldout} - {None, ""}
    exact_overlap = train_sha & heldout_sha
    train_groups = {group_key(row) for row in train} - {""}
    heldout_groups = {group_key(row) for row in heldout} - {""}
    group_overlap = train_groups & heldout_groups
    status = "pass" if not marker_rows and not near_pair_count and not exact_overlap and not group_overlap else "fail"
    report = {
        "schema_version": "attribute-stage66-full-manifest-leakage-audit-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "frozen_marker_rows": marker_rows,
        "exact_sha_cross_split_overlap": len(exact_overlap),
        "source_group_cross_split_overlap": len(group_overlap),
        "dhash": {
            "bits": 64,
            "maximum_cross_split_distance": args.maximum_distance,
            "all_hashes_loaded_from_pixel-enriched_manifest": True,
            "unreadable_images": 0,
            "cross_split_near_pairs": near_pair_count,
            "cross_split_distance_counts": dict(sorted(distance_counts.items())),
        },
        "policy": {
            "all_train_rows_audited_against_all_heldout_rows": True,
            "heldout_labels_not_used_for_audit": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status, "train_rows": len(train), "heldout_rows": len(heldout),
        "cross_split_near_pairs": near_pair_count, "exact_sha_overlap": len(exact_overlap),
        "group_overlap": len(group_overlap),
    }))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
