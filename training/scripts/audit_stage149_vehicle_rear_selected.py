#!/usr/bin/env python3
"""Audit selectively extracted Vehicle-Rear crops for research-only color training.

The sealed future holdout is never opened. Validation rows are inspected first;
train rows that collide exactly or perceptually with validation are excluded.
Near-duplicate frames are allowed only inside the same vehicle track because they
are required for the explicit multi-frame consistency objective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat


BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48")
EXACT_COLORS = {"black", "blue", "brown", "gray", "green", "other", "red", "silver", "white", "yellow"}
PARTIAL_COLORS = {"brown_beige", "yellow_orange"}
OUTPUT_FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "color",
    "color_supervised", "color_supervision", "source_color", "source_dataset",
    "source_license", "license_train_eligible", "formal_train_eligible",
    "research_only", "review_status", "review_method", "camera_id", "video_id",
    "track_group", "source_frame_id", "crop_quality", "lighting", "night",
    "blur", "width", "height", "gray_mean", "gray_stddev", "edge_variance",
    "sha256", "dhash64", "sample_weight", "stage149_origin",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    result = 0
    for bit in (values[:, :-1] > values[:, 1:]).ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if min(values.shape) > 4:
        values = values[2:-2, 2:-2]
    return float(values.var())


def inspect_image(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    if min(width, height) < 32:
        raise ValueError("small_dimension")
    if width / height < 0.35 or width / height > 4.5:
        raise ValueError("extreme_aspect")
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    if mean < 8.0 or mean > 248.0:
        raise ValueError("extreme_exposure")
    quality = "good" if min(width, height) >= 128 and stddev >= 18.0 and edges >= 60.0 else "usable"
    return {
        "sha256": digest,
        "dhash": dhash64(image),
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "crop_quality": quality,
        "lighting": "low_luminance_proxy" if mean < 70.0 else "unknown",
        "blur": edges < 60.0,
    }


def block_values(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


def near_pairs(values: list[int], maximum_distance: int = 4) -> list[tuple[int, int, int]]:
    if maximum_distance != 4:
        raise ValueError("five-block index is exact only for distance four")
    indexes = [defaultdict(list) for _ in BLOCKS]
    pairs: list[tuple[int, int, int]] = []
    for index, value in enumerate(values):
        candidates: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(indexes[block_index].get(block, ()))
        for prior in candidates:
            distance = (value ^ values[prior]).bit_count()
            if distance <= maximum_distance:
                pairs.append((prior, index, distance))
        for block_index, block in enumerate(block_values(value)):
            indexes[block_index][block].append(index)
    return pairs


def sample_weight(color: str, lighting: str, quality: str) -> float:
    weight = 1.0
    if color in {"blue", "brown", "green", "other", "yellow", "brown_beige", "yellow_orange"}:
        weight *= 2.0
    if lighting == "low_luminance_proxy":
        weight *= 1.5
    if quality == "usable":
        weight *= 0.75
    return round(weight, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--official-readme", type=Path, required=True)
    parser.add_argument("--official-license", type=Path, required=True)
    parser.add_argument("--expected-readme-sha256", required=True)
    parser.add_argument("--expected-license-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--minimum-train-rows", type=int, default=4000)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage149 audit evidence")
    for path in (args.selection_manifest, args.selected_root, args.official_readme, args.official_license):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(args.official_readme) != args.expected_readme_sha256.lower():
        raise RuntimeError("official README SHA256 mismatch")
    if sha256_file(args.official_license) != args.expected_license_sha256.lower():
        raise RuntimeError("official LICENSE SHA256 mismatch")
    readme = args.official_readme.read_text(encoding="utf-8", errors="replace")
    license_text = args.official_license.read_text(encoding="utf-8", errors="replace")
    if "vehicle-reid/data.tgz" not in readme or "Vehicle-Rear" not in readme:
        raise RuntimeError("official README does not bind the dataset download")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("official repository Apache-2.0 evidence is missing")

    with args.selection_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    future = [row for row in source_rows if row.get("split") == "future_holdout"]
    candidates = [row for row in source_rows if row.get("split") in {"train", "validation"}]
    if not future or not candidates:
        raise RuntimeError("selection manifest is missing train/validation or sealed holdout rows")
    for row in source_rows:
        text = " ".join(str(value).lower() for value in row.values())
        if any(marker in text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found")

    # Validation is inspected first so training loses every cross-split collision.
    candidates.sort(key=lambda row: (0 if row["split"] == "validation" else 1, row["track_group"], row["tar_member"]))

    def inspect_row(row: dict[str, str]):
        # The selection plan predates the final transfer strategy. Bind every
        # row to the explicitly supplied audited extraction root, never to a
        # stale absolute path embedded in the planning CSV.
        path = (args.selected_root / row["tar_member"]).resolve()
        try:
            return path, inspect_image(path), None
        except Exception as exc:
            return path, None, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(executor.map(inspect_row, candidates))
    rejection = Counter()
    prepared: list[dict[str, object]] = []
    for source, (path, metrics, error) in zip(candidates, inspected):
        if error or metrics is None:
            rejection[f"decode_or_quality:{error}"] += 1
            continue
        color = source.get("color", "unknown")
        supervision = source.get("color_supervision", "")
        if color not in EXACT_COLORS | PARTIAL_COLORS or supervision == "unmapped_fail_closed":
            rejection["unmapped_color"] += 1
            continue
        prepared.append({"source": source, "path": path, "metrics": metrics})

    exact_seen: dict[str, int] = {}
    duplicate_indexes: set[int] = set()
    for index, item in enumerate(prepared):
        digest = str(item["metrics"]["sha256"])
        prior = exact_seen.get(digest)
        if prior is not None:
            duplicate_indexes.add(index)
            rejection["exact_duplicate"] += 1
        else:
            exact_seen[digest] = index

    active_indexes = [index for index in range(len(prepared)) if index not in duplicate_indexes]
    values = [int(prepared[index]["metrics"]["dhash"]) for index in active_indexes]
    perceptual_rejected: set[int] = set()
    cross_split_near = 0
    cross_group_near = 0
    for left_pos, right_pos, _distance in near_pairs(values):
        left_index, right_index = active_indexes[left_pos], active_indexes[right_pos]
        left = prepared[left_index]["source"]
        right = prepared[right_index]["source"]
        if left["track_group"] == right["track_group"]:
            continue
        cross_group_near += 1
        if left["split"] != right["split"]:
            cross_split_near += 1
        # Deterministic order puts validation before train, so the later row is safe to remove.
        perceptual_rejected.add(right_index)

    accepted = []
    for index, item in enumerate(prepared):
        if index in duplicate_indexes or index in perceptual_rejected:
            if index in perceptual_rejected:
                rejection["cross_group_perceptual_near_duplicate"] += 1
            continue
        source = item["source"]
        metrics = item["metrics"]
        accepted.append({
            "image_path": str(item["path"]),
            "split": source["split"],
            "body_type": "unknown",
            "body_type_supervised": "false",
            "color": source["color"],
            "color_supervised": "true",
            "color_supervision": source["color_supervision"],
            "source_color": source["source_color"],
            "source_dataset": "Vehicle-Rear",
            "source_license": "Apache-2.0 official-project scope; research-only isolation",
            "license_train_eligible": "true",
            "formal_train_eligible": "true" if source["split"] == "train" else "false",
            "research_only": "true",
            "review_status": "approved_research_only" if source["split"] == "train" else "locked_validation",
            "review_method": "official_registration_color+decode+quality+sha256+dhash_group_dedup",
            "camera_id": source["camera_id"],
            "video_id": source["video_id"],
            "track_group": source["track_group"],
            "source_frame_id": Path(source["tar_member"]).stem,
            "crop_quality": str(metrics["crop_quality"]),
            "lighting": str(metrics["lighting"]),
            "night": "unknown",
            "blur": str(metrics["blur"]).lower(),
            "width": str(metrics["width"]),
            "height": str(metrics["height"]),
            "gray_mean": str(metrics["gray_mean"]),
            "gray_stddev": str(metrics["gray_stddev"]),
            "edge_variance": str(metrics["edge_variance"]),
            "sha256": str(metrics["sha256"]),
            "dhash64": f"{int(metrics['dhash']):016x}",
            "sample_weight": str(sample_weight(source["color"], str(metrics["lighting"]), str(metrics["crop_quality"]))),
            "stage149_origin": "vehicle_rear_official_selective_r3",
        })

    train = [row for row in accepted if row["split"] == "train"]
    validation = [row for row in accepted if row["split"] == "validation"]
    train_groups = {row["track_group"] for row in train}
    validation_groups = {row["track_group"] for row in validation}
    exact_overlap = {row["sha256"] for row in train} & {row["sha256"] for row in validation}
    train_dhash = [int(row["dhash64"], 16) for row in train]
    validation_dhash = [int(row["dhash64"], 16) for row in validation]
    # Exact five-block cross-split check, excluding no rows because groups are disjoint.
    indexes = [defaultdict(list) for _ in BLOCKS]
    for index, value in enumerate(validation_dhash):
        for block_index, block in enumerate(block_values(value)):
            indexes[block_index][block].append(index)
    residual_near = 0
    for value in train_dhash:
        candidates_idx: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates_idx.update(indexes[block_index].get(block, ()))
        residual_near += sum((value ^ validation_dhash[index]).bit_count() <= 4 for index in candidates_idx)

    dominant = sum(1 for row in train if row["color"] in {"white", "gray", "silver"})
    low_light = sum(1 for row in train if row["lighting"] == "low_luminance_proxy")
    failures = []
    if len(train) < args.minimum_train_rows:
        failures.append(f"train rows {len(train)} < {args.minimum_train_rows}")
    if train_groups & validation_groups:
        failures.append("vehicle groups cross train/validation")
    if exact_overlap:
        failures.append("exact SHA overlap remains")
    if residual_near:
        failures.append("perceptual near-duplicate overlap remains")
    if train and dominant / len(train) > 0.65:
        failures.append("white/gray/silver exceed 65 percent")

    report = {
        "schema_version": "stage149-vehicle-rear-selected-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_only" if not failures else "fail_closed",
        "source": {
            "official_repository": "https://github.com/icarofua/vehicle-rear",
            "official_dataset_url": "https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz",
            "readme_sha256": sha256_file(args.official_readme),
            "license_sha256": sha256_file(args.official_license),
            "license_scope": "official project repository Apache-2.0; admitted research-only and non-deployable",
        },
        "inputs": {
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "selected_root": str(args.selected_root.resolve()),
            "candidate_rows": len(candidates),
            "sealed_future_holdout_rows": len(future),
        },
        "output": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "color_counts_train": dict(sorted(Counter(row["color"] for row in train).items())),
            "color_counts_validation": dict(sorted(Counter(row["color"] for row in validation).items())),
            "low_luminance_proxy_train": low_light,
            "low_luminance_proxy_fraction": low_light / len(train) if train else 0.0,
            "white_gray_silver_fraction": dominant / len(train) if train else 0.0,
            "rejections": dict(sorted(rejection.items())),
        },
        "integrity": {
            "train_validation_group_overlap": len(train_groups & validation_groups),
            "train_validation_exact_sha_overlap": len(exact_overlap),
            "train_validation_dhash_distance_le_4_pairs": residual_near,
            "cross_group_near_pairs_observed_before_filter": cross_group_near,
            "cross_split_near_pairs_observed_before_filter": cross_split_near,
        },
        "policy": {
            "future_holdout_images_opened": False,
            "future_holdout_used_for_selection": False,
            "stage148_test_reused": False,
            "within_track_multiframe_near_images_retained": True,
            "low_luminance_is_proxy_not_verified_night_truth": True,
            "body_labels_fabricated": False,
            "unmapped_colors_forced": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable",
        },
        "failures": failures,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(accepted)
    report["output"]["manifest"] = str(args.output_manifest.resolve())
    report["output"]["manifest_sha256"] = sha256_file(args.output_manifest)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "train_rows": len(train), "validation_rows": len(validation), "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
