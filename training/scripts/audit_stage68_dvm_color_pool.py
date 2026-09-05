#!/usr/bin/env python3
"""Fail-closed audit of the Stage68 DVM color pool and held-out leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")
BLOCK_WIDTHS = (13, 13, 13, 13, 12)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha(row: dict[str, str]) -> str:
    return (row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def dhash(row: dict[str, str]) -> int:
    value = (row.get("dhash64") or "").strip().lower()
    if len(value) != 16:
        raise RuntimeError("missing or invalid 64-bit dHash")
    return int(value, 16)


def block_values(value: int) -> tuple[int, ...]:
    output: list[int] = []
    shift = 0
    for width in BLOCK_WIDTHS:
        output.append((value >> shift) & ((1 << width) - 1))
        shift += width
    return tuple(output)


def near_pair_summary(left: list[int], right: list[int], maximum_distance: int) -> tuple[int, int, dict[int, int]]:
    if maximum_distance != 4:
        raise ValueError("the five-block exact lookup supports maximum_distance=4")
    indexes = [defaultdict(list) for _ in BLOCK_WIDTHS]
    for index, value in enumerate(right):
        for block_index, block in enumerate(block_values(value)):
            indexes[block_index][block].append(index)
    pairs = 0
    matched_left = 0
    distances: Counter[int] = Counter()
    for value in left:
        candidates: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(indexes[block_index].get(block, ()))
        found = False
        for index in candidates:
            distance = (value ^ right[index]).bit_count()
            if distance <= maximum_distance:
                pairs += 1
                found = True
                distances[distance] += 1
        matched_left += int(found)
    return pairs, matched_left, dict(sorted(distances.items()))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dvm-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-rows", type=int, default=118338)
    parser.add_argument("--maximum-distance", type=int, default=4)
    parser.add_argument("--verify-file-sha", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage68 audit evidence")

    rows = load_rows(args.dvm_manifest)
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    unexpected_splits = sorted({row.get("split", "") for row in rows} - {"train", "validation"})
    frozen_markers = sum(
        any(marker in " ".join(str(value).lower() for value in row.values()) for marker in FROZEN_MARKERS)
        for row in rows
    )
    train_groups = {row.get("source_group", "") for row in train} - {""}
    validation_groups = {row.get("source_group", "") for row in validation} - {""}
    train_sha = [pixel_sha(row) for row in train]
    validation_sha = [pixel_sha(row) for row in validation]
    missing_sha = sum(not value for value in train_sha + validation_sha)
    internal_exact_overlap = set(train_sha) & set(validation_sha) - {""}

    heldout_rows = []
    with args.heldout_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") in {"validation", "test"}:
                heldout_rows.append({
                    "split": row.get("split", ""),
                    "sha": pixel_sha(row),
                    "dhash": row.get("dhash64", ""),
                })
    heldout_sha = {row["sha"] for row in heldout_rows if row["sha"]}
    external_exact_overlap = (set(train_sha) - {""}) & heldout_sha
    near_pairs, near_train_rows, near_distances = near_pair_summary(
        [dhash(row) for row in train],
        [int(row["dhash"], 16) for row in heldout_rows],
        args.maximum_distance,
    )

    missing_files = 0
    file_sha_mismatches = 0
    if args.verify_file_sha:
        root = args.dvm_manifest.parent
        for row in rows:
            path = root / row["image_path"]
            if not path.is_file():
                missing_files += 1
            elif file_sha256(path) != pixel_sha(row):
                file_sha_mismatches += 1

    failures = []
    checks = {
        "minimum_train_rows": len(train) >= args.minimum_train_rows,
        "validation_nonempty": bool(validation),
        "only_train_and_validation_splits": not unexpected_splits,
        "frozen_markers_absent": frozen_markers == 0,
        "source_groups_cross_split_disjoint": not (train_groups & validation_groups),
        "manifest_pixel_sha_complete": missing_sha == 0,
        "internal_exact_sha_cross_split_disjoint": not internal_exact_overlap,
        "heldout_exact_sha_disjoint": not external_exact_overlap,
        "heldout_dhash_distance_le_4_disjoint": near_pairs == 0,
        "all_files_present": missing_files == 0,
        "all_file_sha_match": file_sha_mismatches == 0,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    status = "pass" if not failures else "fail"
    report = {
        "schema_version": "attribute-stage68-dvm-color-pool-audit-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dvm_manifest": str(args.dvm_manifest.resolve()),
        "dvm_manifest_sha256": file_sha256(args.dvm_manifest),
        "heldout_manifest": str(args.heldout_manifest.resolve()),
        "heldout_manifest_sha256": file_sha256(args.heldout_manifest),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "heldout_rows_compared": len(heldout_rows),
        "minimum_train_rows": args.minimum_train_rows,
        "unexpected_splits": unexpected_splits,
        "frozen_marker_rows": frozen_markers,
        "source_group_cross_split_overlap": len(train_groups & validation_groups),
        "missing_manifest_pixel_sha": missing_sha,
        "internal_exact_sha_cross_split_overlap": len(internal_exact_overlap),
        "heldout_exact_sha_overlap": len(external_exact_overlap),
        "heldout_dhash": {
            "maximum_distance": args.maximum_distance,
            "near_pairs": near_pairs,
            "matched_train_rows": near_train_rows,
            "distance_counts": near_distances,
        },
        "file_integrity": {
            "full_sha_verification_enabled": args.verify_file_sha,
            "missing_files": missing_files,
            "sha_mismatches": file_sha_mismatches,
        },
        "checks": checks,
        "failures": failures,
        "policy": {
            "heldout_labels_or_predictions_accessed": False,
            "heldout_membership_or_files_modified": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "training_eligibility": "research-only; non-deployable",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "train_rows": len(train), "near_pairs": near_pairs, "failures": failures}))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
