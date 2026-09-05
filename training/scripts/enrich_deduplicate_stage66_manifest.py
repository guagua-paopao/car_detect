#!/usr/bin/env python3
"""Hash every Stage66 crop, remove exact train duplicates, and audit near overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
BOOLEAN_SCENE_FIELDS = ("night", "low_light", "small_target", "occluded", "truncated", "blur")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def supervised(row: dict[str, str], head: str) -> bool:
    value = row.get(f"{head}_supervised", "").strip().lower()
    return True if not value else value in {"1", "true", "yes"}


def dhash64(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def block_values(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


def count_near_pairs(left: list[int], right: list[int], maximum_distance: int = 4) -> tuple[int, Counter]:
    indexes = [defaultdict(list) for _ in BLOCKS]
    for index, value in enumerate(right):
        for block_index, block in enumerate(block_values(value)):
            indexes[block_index][block].append(index)
    count = 0
    distances: Counter = Counter()
    for value in left:
        candidates = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(indexes[block_index].get(block, ()))
        for index in candidates:
            distance = (value ^ right[index]).bit_count()
            if distance <= maximum_distance:
                count += 1
                distances[distance] += 1
    return count, distances


def merge_exact_group(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, bool]]:
    chosen = dict(sorted(rows, key=lambda row: row.get("image_path", ""))[0])
    conflicts: dict[str, bool] = {}
    for head, field in (("body_type", "body_type"), ("color", "color")):
        labels = {
            row.get(field, "") for row in rows
            if supervised(row, head) and row.get(field, "") not in {"", "unknown"}
        }
        conflicts[head] = len(labels) > 1
        if len(labels) == 1:
            chosen[field] = next(iter(labels))
            chosen[f"{head}_supervised"] = "true"
        elif len(labels) > 1:
            chosen[field] = "unknown"
            chosen[f"{head}_supervised"] = "false"
    families = {row.get("coarse_body_family", "") for row in rows} - {""}
    conflicts["coarse_body_family"] = len(families) > 1
    chosen["coarse_body_family"] = next(iter(families)) if len(families) == 1 else ""
    for field in BOOLEAN_SCENE_FIELDS:
        if any(truthy(row.get(field)) for row in rows):
            chosen[field] = "true"
    weights = []
    for row in rows:
        try:
            weights.append(float(row.get("sample_weight") or 1.0))
        except ValueError:
            weights.append(1.0)
    chosen["sample_weight"] = f"{max(weights):.6f}"
    chosen["stage66_exact_dedup_count"] = str(len(rows))
    chosen["stage66_exact_dedup_conflict"] = "true" if any(conflicts.values()) else "false"
    return chosen, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage66 hash/dedup evidence")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("manifest has no header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    if any(any(marker in " ".join(str(v).lower() for v in row.values()) for marker in FROZEN_MARKERS) for row in rows):
        raise RuntimeError("frozen marker found")

    def image_path(row: dict[str, str]) -> Path:
        path = Path(row["image_path"])
        return path if path.is_absolute() else (args.manifest.parent / path).resolve()

    unique_paths: dict[str, Path] = {}
    for row in rows:
        unique_paths[row["image_path"]] = image_path(row)

    def resolve(item: tuple[str, Path]) -> tuple[str, str | None, str | None, str | None]:
        key, path = item
        try:
            return key, sha256(path), dhash64(path), None
        except Exception as error:
            return key, None, None, f"{path}: {error}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        resolved = list(executor.map(resolve, unique_paths.items()))
    errors = [error for _, _, _, error in resolved if error]
    if errors:
        raise RuntimeError(f"failed to hash {len(errors)} crops; first={errors[0]}")
    hashes = {key: (file_hash, perceptual) for key, file_hash, perceptual, _ in resolved}
    for row in rows:
        file_hash, perceptual = hashes[row["image_path"]]
        row["crop_sha256"] = str(file_hash)
        row["dhash64"] = str(perceptual)

    heldout = [row for row in rows if row.get("split") in {"validation", "test"}]
    train = [row for row in rows if row.get("split") == "train"]
    heldout_hashes = {row["crop_sha256"] for row in heldout}
    exact_cross_split_train = [row for row in train if row["crop_sha256"] in heldout_hashes]
    train = [row for row in train if row["crop_sha256"] not in heldout_hashes]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train:
        groups[row["crop_sha256"]].append(row)
    merged_train: list[dict[str, str]] = []
    exact_duplicate_groups = 0
    exact_duplicate_rows_removed = 0
    conflict_counts = Counter()
    for group in groups.values():
        if len(group) == 1:
            row = dict(group[0])
            row["stage66_exact_dedup_count"] = "1"
            row["stage66_exact_dedup_conflict"] = "false"
            merged_train.append(row)
            continue
        exact_duplicate_groups += 1
        exact_duplicate_rows_removed += len(group) - 1
        merged, conflicts = merge_exact_group(group)
        merged_train.append(merged)
        conflict_counts.update(key for key, value in conflicts.items() if value)
    merged_train.sort(key=lambda row: row.get("image_path", ""))
    output = merged_train + [dict(row) for row in rows if row.get("split") != "train"]
    original_eval = [(row.get("split"), row.get("image_path")) for row in rows if row.get("split") != "train"]
    output_eval = [(row.get("split"), row.get("image_path")) for row in output if row.get("split") != "train"]
    if original_eval != output_eval:
        raise RuntimeError("heldout membership or order changed")
    train_dhash = [int(row["dhash64"], 16) for row in merged_train]
    heldout_dhash = [int(row["dhash64"], 16) for row in heldout]
    near_count, near_distances = count_near_pairs(train_dhash, heldout_dhash, 4)
    output_fields = list(dict.fromkeys([*fields, "stage66_exact_dedup_count", "stage66_exact_dedup_conflict"]))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    report = {
        "schema_version": "attribute-stage66-hash-dedup-report-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "input_rows": len(rows),
        "unique_image_paths_hashed": len(unique_paths),
        "unreadable_images": 0,
        "exact_cross_split_train_rows_removed": len(exact_cross_split_train),
        "exact_duplicate_train_groups": exact_duplicate_groups,
        "exact_duplicate_train_rows_removed": exact_duplicate_rows_removed,
        "exact_duplicate_label_conflicts": dict(conflict_counts),
        "output_rows": len(output),
        "output_train_rows": len(merged_train),
        "heldout_rows_preserved": len(heldout),
        "train_to_heldout_dhash_distance_le_4_pairs_diagnostic": near_count,
        "train_to_heldout_dhash_distance_counts": dict(sorted(near_distances.items())),
        "policy": {
            "exact_byte_duplicates_removed_from_train": True,
            "exact_cross_split_train_duplicates_removed": True,
            "perceptual_similarity_is_diagnostic_not_automatic_identity": True,
            "heldout_membership_preserved": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "input_rows", "unique_image_paths_hashed", "exact_cross_split_train_rows_removed",
        "exact_duplicate_train_rows_removed", "output_rows", "train_to_heldout_dhash_distance_le_4_pairs_diagnostic",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
