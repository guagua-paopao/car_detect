#!/usr/bin/env python3
"""Remove Stage66 train crops perceptually near any held-out crop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dhash(row: dict[str, str]) -> int:
    value = str(row.get("dhash64", "")).strip().lower()
    if len(value) != 16:
        raise ValueError(f"invalid dHash for {row.get('image_path')}: {value!r}")
    return int(value, 16)


def block_values(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


def near_matches(left: list[int], right: list[int], maximum_distance: int) -> list[list[tuple[int, int]]]:
    if maximum_distance != 4:
        raise ValueError("five-block exact lookup requires maximum_distance=4")
    indexes = [defaultdict(list) for _ in BLOCKS]
    for right_index, value in enumerate(right):
        for block_index, block in enumerate(block_values(value)):
            indexes[block_index][block].append(right_index)
    result: list[list[tuple[int, int]]] = []
    for value in left:
        candidates: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(indexes[block_index].get(block, ()))
        matches = []
        for right_index in candidates:
            distance = (value ^ right[right_index]).bit_count()
            if distance <= maximum_distance:
                matches.append((right_index, distance))
        result.append(sorted(matches, key=lambda item: (item[1], item[0])))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--removed-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-distance", type=int, default=4)
    args = parser.parse_args()
    for path in (args.output_manifest, args.removed_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage66 near-dedup evidence: {path}")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("manifest has no header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    marker_rows = sum(
        any(marker in " ".join(str(value).lower() for value in row.values()) for marker in FROZEN_MARKERS)
        for row in rows
    )
    if marker_rows:
        raise RuntimeError("frozen-video marker found")
    train = [row for row in rows if row.get("split") == "train"]
    heldout = [row for row in rows if row.get("split") in {"validation", "test"}]
    if not train or not heldout:
        raise RuntimeError("train or held-out split is empty")
    matches = near_matches(
        [parse_dhash(row) for row in train],
        [parse_dhash(row) for row in heldout],
        args.maximum_distance,
    )
    kept_train = []
    removed = []
    pair_distances: Counter[int] = Counter()
    minimum_distances: Counter[int] = Counter()
    for row, row_matches in zip(train, matches):
        if not row_matches:
            kept_train.append(row)
            continue
        item = dict(row)
        item["stage66_near_heldout_min_distance"] = str(row_matches[0][1])
        item["stage66_near_heldout_pair_count"] = str(len(row_matches))
        removed.append(item)
        minimum_distances[row_matches[0][1]] += 1
        pair_distances.update(distance for _, distance in row_matches)
    output = kept_train + [row for row in rows if row.get("split") != "train"]
    original_heldout = [(row.get("split"), row.get("image_path")) for row in rows if row.get("split") != "train"]
    output_heldout = [(row.get("split"), row.get("image_path")) for row in output if row.get("split") != "train"]
    if original_heldout != output_heldout:
        raise RuntimeError("held-out membership or order changed")
    residual_matches = near_matches(
        [parse_dhash(row) for row in kept_train],
        [parse_dhash(row) for row in heldout],
        args.maximum_distance,
    )
    residual_pairs = sum(len(item) for item in residual_matches)
    train_sha = {row.get("crop_sha256") for row in kept_train} - {None, ""}
    heldout_sha = {row.get("crop_sha256") for row in heldout} - {None, ""}
    exact_overlap = len(train_sha & heldout_sha)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    removed_fields = [*fields, "stage66_near_heldout_min_distance", "stage66_near_heldout_pair_count"]
    with args.removed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=removed_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(removed)
    status = "pass" if residual_pairs == 0 and exact_overlap == 0 else "fail"
    report = {
        "schema_version": "attribute-stage66-cross-split-near-dedup-report-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "removed_manifest": str(args.removed_manifest.resolve()),
        "removed_manifest_sha256": sha256(args.removed_manifest),
        "maximum_hamming_distance": args.maximum_distance,
        "input_rows": len(rows),
        "input_train_rows": len(train),
        "heldout_rows_preserved": len(heldout),
        "removed_train_rows": len(removed),
        "removed_cross_split_pairs": sum(pair_distances.values()),
        "removed_pair_distance_counts": dict(sorted(pair_distances.items())),
        "removed_train_minimum_distance_counts": dict(sorted(minimum_distances.items())),
        "output_rows": len(output),
        "output_train_rows": len(kept_train),
        "residual_cross_split_near_pairs": residual_pairs,
        "residual_exact_sha_cross_split_overlap": exact_overlap,
        "policy": {
            "all_train_rows_with_dhash_distance_le_4_to_heldout_removed": True,
            "heldout_membership_preserved": True,
            "heldout_labels_not_used_for_filtering": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status, "removed_train_rows": len(removed),
        "removed_cross_split_pairs": sum(pair_distances.values()),
        "output_train_rows": len(kept_train), "residual_cross_split_near_pairs": residual_pairs,
    }))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
