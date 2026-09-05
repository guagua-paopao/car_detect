#!/usr/bin/env python3
"""Build a quality-filtered VCAS body-type dataset from official MIO-TCD labels.

MIO-TCD does not label color and its generic ``car`` class does not distinguish
sedan/SUV/MPV.  This importer therefore supervises only body types that follow
directly from the official class names.  The color head is explicitly masked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


SOURCE_TO_BODY = {
    "articulated_truck": "heavy_truck",
    "bus": "bus",
    "pickup_truck": "pickup",
    "single_unit_truck": "light_truck",
    "work_van": "van",
}
SPLITS = ("train", "validation", "test")
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
    "source_label",
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
    parser.add_argument("--train-per-class", type=int, default=6000)
    parser.add_argument("--validation-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--oversample-factor", type=int, default=5)
    parser.add_argument("--min-dimension", type=int, default=64)
    parser.add_argument("--min-gray-stddev", type=float, default=12.0)
    parser.add_argument("--min-edge-variance", type=float, default=35.0)
    parser.add_argument("--max-dhash-distance", type=int, default=4)
    parser.add_argument("--dataset-version", default="dataset-large-v1-mio-attributes")
    parser.add_argument("--license-review-ref", required=True)
    return parser.parse_args()


def read_labels(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(archive_path: Path) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    with tarfile.open(archive_path, "r:") as archive:
        member = archive.getmember("gt_train.csv")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("gt_train.csv cannot be read from archive")
        text = io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="")
        for row in csv.reader(text):
            if len(row) != 2:
                continue
            image_id, source_label = (value.strip() for value in row)
            if source_label in SOURCE_TO_BODY:
                by_label[source_label].append(image_id)
    return by_label


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    values = np.asarray(gray, dtype=np.float32)
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
        for existing, image_id, source_label in candidates:
            if (existing ^ value).bit_count() <= self.maximum_distance:
                return image_id, source_label
        return None

    def add(self, value: int, image_id: str, source_label: str) -> None:
        item = (value, image_id, source_label)
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
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    if mean < 12.0 or mean > 245.0:
        raise ValueError("extreme_exposure")
    if stddev < args.min_gray_stddev:
        raise ValueError("low_contrast")
    if edges < args.min_edge_variance:
        raise ValueError("blurred")
    quality = "good" if min(width, height) >= 96 and stddev >= 20.0 and edges >= 80.0 else "usable"
    return image, {
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "crop_quality": quality,
        "night": mean < 55.0,
        "blur": edges < 80.0,
        "dhash": dhash64(image),
    }


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    labels = read_labels(args.labels.resolve())
    body_types = list(labels["body_types"])
    missing = sorted(set(SOURCE_TO_BODY.values()) - set(body_types))
    if missing:
        raise RuntimeError(f"MIO mapping is outside the local label contract: {missing}")
    if not args.license_review_ref.strip():
        raise ValueError("license review reference is required")

    requested = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    target_per_class = sum(requested.values())
    ground_truth = load_ground_truth(archive_path)
    rng = random.Random(args.seed)
    needed: dict[str, tuple[str, str]] = {}
    source_counts = {name: len(values) for name, values in ground_truth.items()}
    for source_label, image_ids in sorted(ground_truth.items()):
        rng.shuffle(image_ids)
        candidate_count = min(len(image_ids), target_per_class * args.oversample_factor)
        for image_id in image_ids[:candidate_count]:
            needed[f"train/{source_label}/{image_id}.jpg"] = (image_id, source_label)

    output_root.mkdir(parents=True)
    staging_root = output_root / "_selected"
    staging_root.mkdir()
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    exact_hashes: dict[str, tuple[str, str]] = {}
    near_index = NearDuplicateIndex(args.max_dhash_distance)

    with tarfile.open(archive_path, "r|") as archive:
        for member in archive:
            identity = needed.get(member.name)
            if identity is None or not member.isfile():
                continue
            image_id, source_label = identity
            target_label = SOURCE_TO_BODY[source_label]
            if len(accepted[target_label]) >= target_per_class:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                counters["missing_payload"] += 1
                continue
            payload = extracted.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in exact_hashes:
                counters["rejected_exact_duplicate"] += 1
                continue
            try:
                image, metrics = inspect_image(payload, args)
            except Exception as exc:
                counters[f"rejected_{str(exc)}"] += 1
                continue
            near_match = near_index.match(metrics["dhash"])
            if near_match is not None:
                counters["rejected_near_duplicate"] += 1
                if near_match[1] != source_label:
                    counters["rejected_near_duplicate_label_conflict"] += 1
                continue
            exact_hashes[digest] = (image_id, source_label)
            near_index.add(metrics["dhash"], image_id, source_label)
            staged_path = staging_root / f"mio_{image_id}.jpg"
            image.save(staged_path, format="JPEG", quality=95, optimize=True)
            accepted[target_label].append(
                {
                    "image_id": image_id,
                    "source_label": source_label,
                    "body_type": target_label,
                    "staged_path": staged_path,
                    "sha256": digest,
                    **metrics,
                }
            )
            counters[f"accepted_{target_label}"] += 1

    shortages = {
        body_type: target_per_class - len(rows)
        for body_type, rows in accepted.items()
        if len(rows) < target_per_class
    }
    expected_types = set(SOURCE_TO_BODY.values())
    for body_type in expected_types - set(accepted):
        shortages[body_type] = target_per_class
    if shortages:
        raise RuntimeError(f"not enough quality samples after audit: {shortages}")

    manifest_rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    for body_type in sorted(expected_types, key=body_types.index):
        rows = accepted[body_type]
        random.Random(args.seed + body_types.index(body_type) * 1009).shuffle(rows)
        offset = 0
        for split in SPLITS:
            split_rows = rows[offset : offset + requested[split]]
            offset += requested[split]
            crop_dir = output_root / "crops" / split
            crop_dir.mkdir(parents=True, exist_ok=True)
            for item in split_rows:
                image_id = item["image_id"]
                target_path = crop_dir / f"mio_{image_id}.jpg"
                os.replace(item["staged_path"], target_path)
                manifest_rows.append(
                    {
                        "image_path": target_path.relative_to(output_root).as_posix(),
                        "body_type": body_type,
                        "color": "unknown",
                        "crop_quality": item["crop_quality"],
                        "viewpoint": "unknown",
                        "blur": str(item["blur"]).lower(),
                        "occluded": "false",
                        "truncated": "false",
                        "night": str(item["night"]).lower(),
                        "camera_id": f"mio_unknown_camera_{image_id}",
                        "video_id": f"mio_unknown_video_{image_id}",
                        "track_group": f"mio_image_{image_id}",
                        "split": split,
                        "source_frame_id": image_id,
                        "review_status": "approved",
                        "body_type_supervised": "true",
                        "color_supervised": "false",
                        "source_dataset": "MIO-TCD-Classification-2017",
                        "source_label": item["source_label"],
                        "source_license": "CC-BY-NC-SA-4.0",
                        "review_method": "official_label+decode+quality+dhash_dedup",
                        "sha256": item["sha256"],
                        "dhash64": f"{item['dhash']:016x}",
                        "width": str(item["width"]),
                        "height": str(item["height"]),
                        "gray_mean": str(item["gray_mean"]),
                        "gray_stddev": str(item["gray_stddev"]),
                        "edge_variance": str(item["edge_variance"]),
                    }
                )
                split_counts[f"{split}:{body_type}"] += 1
    shutil.rmtree(staging_root)

    manifest_path = output_root / "attribute_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    card = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_version": labels["labels_version"],
        "source": {
            "name": "MIO-TCD Classification 2017",
            "url": "https://tcd.miovision.com/challenge/dataset.html",
            "archive_sha256": sha256_file(archive_path),
            "license": "CC-BY-NC-SA-4.0",
            "license_review_ref": args.license_review_ref,
            "restriction": "non-commercial; attribution and share-alike required",
            "official_train_label_counts": source_counts,
        },
        "mapping": SOURCE_TO_BODY,
        "excluded_official_classes": {
            "car": "cannot distinguish sedan/suv/mpv from the official label",
            "motorcycle": "not used to synthesize the body-type other class; existing human-reviewed project samples are retained",
            "bicycle": "outside the project's vehicle attribute contract",
            "non-motorized_vehicle": "outside the project's vehicle attribute contract",
            "pedestrian": "outside the project's vehicle attribute contract",
            "background": "not a vehicle attribute sample",
        },
        "color_policy": "unknown with color_supervised=false; MIO-TCD has no color ground truth",
        "identity_limit": "camera/track identity is unavailable; exact and perceptual duplicates are removed",
        "quality_thresholds": {
            "min_dimension": args.min_dimension,
            "min_gray_stddev": args.min_gray_stddev,
            "min_edge_variance": args.min_edge_variance,
            "max_dhash_distance": args.max_dhash_distance,
        },
        "rows": len(manifest_rows),
        "split_body_counts": dict(split_counts),
        "audit_counters": dict(counters),
    }
    with (output_root / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"PASS: wrote {len(manifest_rows)} audited MIO-TCD samples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
