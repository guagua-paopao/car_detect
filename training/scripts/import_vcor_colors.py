#!/usr/bin/env python3
"""Import VCoR official color folders into the VCAS attribute contract.

Only the color head is supervised.  Body type is deliberately masked because
VCoR provides no body-type ground truth.  Test and validation members are
processed before training members so cross-split duplicates cannot contaminate
the independent evaluation sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


SOURCE_TO_COLOR = {
    "black": "black",
    "white": "white",
    "gray": "silver_gray",
    "grey": "silver_gray",
    "silver": "silver_gray",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow_orange",
    "orange": "yellow_orange",
    "gold": "yellow_orange",
    "brown": "brown_beige",
    "beige": "brown_beige",
    "tan": "brown_beige",
    "purple": "other",
    "pink": "other",
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = [
    "image_path",
    "body_type",
    "color",
    "crop_quality",
    "viewpoint",
    "blur",
    "occluded",
    "truncated",
    "night",
    "camera_id",
    "video_id",
    "track_group",
    "split",
    "source_frame_id",
    "review_status",
    "body_type_supervised",
    "color_supervised",
    "source_dataset",
    "source_color",
    "source_license",
    "review_method",
    "sha256",
    "dhash64",
    "width",
    "height",
    "gray_mean",
    "gray_stddev",
    "edge_variance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--min-dimension", type=int, default=64)
    parser.add_argument("--min-gray-stddev", type=float, default=10.0)
    parser.add_argument("--min-edge-variance", type=float, default=25.0)
    parser.add_argument("--max-dhash-distance", type=int, default=4)
    parser.add_argument("--dataset-version", default="dataset-large-v1-vcor-colors")
    parser.add_argument("--license-review-ref", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_labels(name: str) -> tuple[str, str] | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    parts = [part.strip().lower().replace(" ", "_") for part in path.parts]
    splits = [SPLIT_ALIASES[part] for part in parts if part in SPLIT_ALIASES]
    colors = [part for part in parts if part in SOURCE_TO_COLOR]
    if len(set(splits)) != 1 or len(set(colors)) != 1:
        return None
    return splits[0], colors[0]


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if min(values.shape) > 4:
        values = values[2:-2, 2:-2]
    return float(values.var())


class NearDuplicateIndex:
    def __init__(self, maximum_distance: int) -> None:
        self.maximum_distance = maximum_distance
        self.bands: dict[tuple[int, int], list[tuple[int, str, str]]] = defaultdict(list)

    def match(self, value: int) -> tuple[str, str] | None:
        candidates: dict[tuple[int, str, str], None] = {}
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            for item in self.bands.get((band_index, band), []):
                candidates[item] = None
        for existing, member_name, split in candidates:
            if (existing ^ value).bit_count() <= self.maximum_distance:
                return member_name, split
        return None

    def add(self, value: int, member_name: str, split: str) -> None:
        item = (value, member_name, split)
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            self.bands[(band_index, band)].append(item)


def inspect_image(payload: bytes, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    if min(width, height) < args.min_dimension:
        raise ValueError("small_dimension")
    aspect = width / height
    if aspect < 0.45 or aspect > 4.0:
        raise ValueError("extreme_aspect")
    stats = ImageStat.Stat(image.convert("L"))
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    if mean < 12.0 or mean > 245.0:
        raise ValueError("extreme_exposure")
    if stddev < args.min_gray_stddev:
        raise ValueError("low_contrast")
    if edges < args.min_edge_variance:
        raise ValueError("blurred")
    quality = "good" if min(width, height) >= 96 and stddev >= 18.0 and edges >= 70.0 else "usable"
    return image, {
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "crop_quality": quality,
        "night": mean < 55.0,
        "blur": edges < 70.0,
        "dhash": dhash64(image),
    }


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    with args.labels.resolve().open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    missing = sorted(set(SOURCE_TO_COLOR.values()) - set(labels["colors"]))
    if missing:
        raise RuntimeError(f"VCoR mapping is outside the local color contract: {missing}")

    with zipfile.ZipFile(archive_path) as archive:
        candidates: list[tuple[int, str, str, str]] = []
        unmatched_images = 0
        for info in archive.infolist():
            inferred = infer_labels(info.filename)
            if inferred is None:
                if PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                    unmatched_images += 1
                continue
            split, source_color = inferred
            split_priority = {"test": 0, "validation": 1, "train": 2}[split]
            candidates.append((split_priority, info.filename, split, source_color))
        random.Random(args.seed).shuffle(candidates)
        candidates.sort(key=lambda item: item[0])

        output_root.mkdir(parents=True)
        exact_hashes: dict[str, tuple[str, str]] = {}
        near_index = NearDuplicateIndex(args.max_dhash_distance)
        counters: Counter[str] = Counter()
        rows: list[dict[str, str]] = []
        for _, member_name, split, source_color in candidates:
            try:
                payload = archive.read(member_name)
            except Exception:
                counters["rejected_unreadable_member"] += 1
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest in exact_hashes:
                counters["rejected_exact_duplicate"] += 1
                if exact_hashes[digest][1] != source_color:
                    counters["rejected_exact_duplicate_label_conflict"] += 1
                continue
            try:
                image, metrics = inspect_image(payload, args)
            except Exception as exc:
                counters[f"rejected_{str(exc)}"] += 1
                continue
            near_match = near_index.match(metrics["dhash"])
            if near_match is not None:
                counters["rejected_near_duplicate"] += 1
                if near_match[1] != split:
                    counters["rejected_cross_split_near_duplicate"] += 1
                continue
            exact_hashes[digest] = (member_name, source_color)
            near_index.add(metrics["dhash"], member_name, split)
            canonical_color = SOURCE_TO_COLOR[source_color]
            sample_key = hashlib.sha1(member_name.encode("utf-8")).hexdigest()[:16]
            target_dir = output_root / "crops" / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"vcor_{sample_key}.jpg"
            image.save(target_path, format="JPEG", quality=95, optimize=True)
            rows.append(
                {
                    "image_path": target_path.relative_to(output_root).as_posix(),
                    "body_type": "unknown",
                    "color": canonical_color,
                    "crop_quality": metrics["crop_quality"],
                    "viewpoint": "unknown",
                    "blur": str(metrics["blur"]).lower(),
                    "occluded": "false",
                    "truncated": "false",
                    "night": str(metrics["night"]).lower(),
                    "camera_id": f"vcor_unknown_camera_{sample_key}",
                    "video_id": f"vcor_unknown_video_{sample_key}",
                    "track_group": f"vcor_image_{sample_key}",
                    "split": split,
                    "source_frame_id": member_name,
                    "review_status": "approved",
                    "body_type_supervised": "false",
                    "color_supervised": "true",
                    "source_dataset": "VCoR",
                    "source_color": source_color,
                    "source_license": "research-and-education-only; original-authors",
                    "review_method": "official_folder+decode+quality+dhash_dedup",
                    "sha256": digest,
                    "dhash64": f"{metrics['dhash']:016x}",
                    "width": str(metrics["width"]),
                    "height": str(metrics["height"]),
                    "gray_mean": str(metrics["gray_mean"]),
                    "gray_stddev": str(metrics["gray_stddev"]),
                    "edge_variance": str(metrics["edge_variance"]),
                }
            )
            counters[f"accepted_{split}_{canonical_color}"] += 1

    if not rows:
        raise RuntimeError("no VCoR images matched a split/color folder and passed audit")
    split_color_counts = Counter(f"{row['split']}:{row['color']}" for row in rows)
    for split in ("validation", "test"):
        missing_eval = sorted(
            color for color in set(SOURCE_TO_COLOR.values()) if split_color_counts[f"{split}:{color}"] == 0
        )
        if missing_eval:
            raise RuntimeError(f"{split} is missing canonical colors after audit: {missing_eval}")
    manifest_path = output_root / "attribute_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    card = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_version": labels["labels_version"],
        "source": {
            "name": "VCoR Vehicle Color Recognition Dataset",
            "url": "https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset",
            "archive_sha256": sha256_file(archive_path),
            "license": "research-and-education-only; data files copyright original authors",
            "license_review_ref": args.license_review_ref,
        },
        "mapping": SOURCE_TO_COLOR,
        "body_type_policy": "unknown with body_type_supervised=false; VCoR has no body-type truth",
        "split_policy": "preserve official folders; process test/validation before train; reject near duplicates globally",
        "unmatched_image_members": unmatched_images,
        "rows": len(rows),
        "split_color_counts": dict(split_color_counts),
        "audit_counters": dict(counters),
    }
    with (output_root / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"PASS: wrote {len(rows)} audited VCoR samples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
