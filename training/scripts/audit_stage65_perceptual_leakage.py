#!/usr/bin/env python3
"""Audit Stage65 added crops against all held-out images using 64-bit dHash."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dhash(value: object) -> int | None:
    text = str(value or "").strip().lower()
    if len(text) != 16:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def image_dhash(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return result


def blocks(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


def find_near_pairs(left: list[int], right: list[int], maximum_distance: int = 4) -> list[tuple[int, int, int]]:
    if maximum_distance != 4:
        raise ValueError("the five-block exact index is defined for maximum_distance=4")
    indexes = [defaultdict(list) for _ in BLOCKS]
    for right_index, value in enumerate(right):
        for block_index, block in enumerate(blocks(value)):
            indexes[block_index][block].append(right_index)
    pairs = []
    for left_index, value in enumerate(left):
        candidates = set()
        for block_index, block in enumerate(blocks(value)):
            candidates.update(indexes[block_index].get(block, ()))
        for right_index in candidates:
            distance = (value ^ right[right_index]).bit_count()
            if distance <= maximum_distance:
                pairs.append((left_index, right_index, distance))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage65 leakage evidence")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    appended = [row for row in rows if str(row.get("adverse_supervised", "")).lower() == "true"]
    heldout = [row for row in rows if row.get("split") in {"validation", "test"}]
    if len(appended) < 4000 or not heldout:
        raise RuntimeError("Stage65 audit inputs are incomplete")
    for row in appended + heldout:
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found in Stage65 leakage audit")

    def path_for(row: dict[str, str]) -> Path:
        path = Path(row["image_path"])
        return path if path.is_absolute() else (args.manifest.parent / path).resolve()

    def resolve_hash(row: dict[str, str]) -> tuple[int | None, bool, str | None]:
        existing = parse_dhash(row.get("dhash64"))
        if existing is not None:
            return existing, False, None
        path = path_for(row)
        try:
            return image_dhash(path), True, None
        except Exception as error:
            return None, True, f"{path}: {error}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        resolved = list(executor.map(resolve_hash, appended + heldout))
    errors = [error for _, _, error in resolved if error]
    if errors:
        raise RuntimeError(f"failed to hash {len(errors)} images; first={errors[0]}")
    values = [value for value, _, _ in resolved]
    if any(value is None for value in values):
        raise RuntimeError("Stage65 dHash resolution is incomplete")
    appended_hashes = [int(value) for value in values[:len(appended)]]
    heldout_hashes = [int(value) for value in values[len(appended):]]
    computed = sum(was_computed for _, was_computed, _ in resolved)

    cross_pairs = find_near_pairs(appended_hashes, heldout_hashes)
    within_raw = find_near_pairs(appended_hashes, appended_hashes)
    within_pairs = [(a, b, distance) for a, b, distance in within_raw if a < b]
    appended_groups = {
        row.get("source_image_id") or row.get("source_frame_id") or row.get("video_id")
        for row in appended
    }
    heldout_groups = {
        row.get("source_image_id") or row.get("source_frame_id") or row.get("video_id")
        for row in heldout
    }
    appended_groups.discard(None); appended_groups.discard("")
    heldout_groups.discard(None); heldout_groups.discard("")
    group_overlap = sorted(appended_groups & heldout_groups)
    appended_sha = {row.get("sha256") or row.get("crop_sha256") for row in appended}
    heldout_sha = {row.get("sha256") or row.get("crop_sha256") for row in heldout}
    appended_sha.discard(None); appended_sha.discard("")
    heldout_sha.discard(None); heldout_sha.discard("")
    sha_overlap = sorted(appended_sha & heldout_sha)
    status = "pass" if not cross_pairs and not group_overlap and not sha_overlap else "fail"
    report = {
        "schema_version": "stage65-perceptual-leakage-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "appended_rows": len(appended), "heldout_rows": len(heldout),
        "dhash": {
            "bits": 64, "maximum_cross_split_distance": 4,
            "existing_hashes": len(resolved) - computed, "computed_hashes": computed,
            "unreadable_images": 0,
            "cross_split_near_pairs": len(cross_pairs),
            "within_appended_near_pairs": len(within_pairs),
            "within_appended_distance_counts": dict(sorted(Counter(distance for _, _, distance in within_pairs).items())),
            "within_appended_cross_source_pairs": sum(
                (appended[a].get("source_image_id") or appended[a].get("source_frame_id"))
                != (appended[b].get("source_image_id") or appended[b].get("source_frame_id"))
                for a, b, _ in within_pairs
            ),
        },
        "exact_sha_cross_split_overlap": len(sha_overlap),
        "source_group_cross_split_overlap": len(group_overlap),
        "policy": {
            "all_missing_heldout_dhash_computed_from_pixels": True,
            "five_block_index_exact_for_hamming_distance_le_4": True,
            "test_images_opened_only_for_leakage_audit_not_model_evaluation": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": status, "cross_pairs": len(cross_pairs), "computed_hashes": computed}, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
