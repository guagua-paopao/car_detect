#!/usr/bin/env python3
"""Import explicit Stanford Cars Sedan/SUV/Minivan classes into VCAS labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from import_vcor_colors import NearDuplicateIndex, inspect_image, sha256_file


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
    "source_class_name",
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
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--min-dimension", type=int, default=96)
    parser.add_argument("--min-gray-stddev", type=float, default=12.0)
    parser.add_argument("--min-edge-variance", type=float, default=35.0)
    parser.add_argument("--max-dhash-distance", type=int, default=4)
    parser.add_argument("--dataset-version", default="dataset-large-v1-stanford-body")
    parser.add_argument("--license-review-ref", required=True)
    return parser.parse_args()


def body_type_from_class_name(class_name: str) -> str | None:
    lowered = class_name.casefold()
    if "minivan" in lowered:
        return "mpv"
    if " suv " in f" {lowered} ":
        return "suv"
    if "sedan" in lowered:
        return "sedan"
    return None


def infer_member(name: str) -> tuple[str, str, str] | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    source_split = next((part.lower() for part in path.parts if part.lower() in {"train", "test"}), None)
    class_name = path.parts[-2] if len(path.parts) >= 2 else ""
    body_type = body_type_from_class_name(class_name)
    if source_split is None or body_type is None:
        return None
    return source_split, class_name, body_type


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not (0.0 < args.validation_ratio < 0.5):
        raise ValueError("--validation-ratio must be between 0 and 0.5")
    with args.labels.resolve().open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not {"sedan", "suv", "mpv"}.issubset(set(labels["body_types"])):
        raise RuntimeError("Stanford mapping is outside the local body-type contract")

    with zipfile.ZipFile(archive_path) as archive:
        train_by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
        test_members: list[tuple[str, str, str]] = []
        unmatched_images = 0
        for info in archive.infolist():
            inferred = infer_member(info.filename)
            if inferred is None:
                if PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                    unmatched_images += 1
                continue
            source_split, class_name, body_type = inferred
            if source_split == "test":
                test_members.append((info.filename, class_name, body_type))
            else:
                train_by_class[class_name].append((info.filename, body_type))
        assigned: list[tuple[int, str, str, str, str]] = []
        for member_name, class_name, body_type in test_members:
            assigned.append((0, member_name, "test", class_name, body_type))
        for class_name, items in sorted(train_by_class.items()):
            class_seed = int(hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:8], 16)
            random.Random(args.seed + class_seed).shuffle(items)
            names = [item[0] for item in items]
            body_type = items[0][1]
            validation_count = max(1, round(len(names) * args.validation_ratio))
            for index, member_name in enumerate(names):
                split = "validation" if index < validation_count else "train"
                priority = 1 if split == "validation" else 2
                assigned.append((priority, member_name, split, class_name, body_type))
        assigned.sort(key=lambda item: (item[0], item[1]))

        output_root.mkdir(parents=True)
        exact_hashes: dict[str, tuple[str, str]] = {}
        near_index = NearDuplicateIndex(args.max_dhash_distance)
        counters: Counter[str] = Counter()
        rows: list[dict[str, str]] = []
        mapped_class_names: dict[str, set[str]] = defaultdict(set)
        for _, member_name, split, class_name, body_type in assigned:
            payload = archive.read(member_name)
            digest = hashlib.sha256(payload).hexdigest()
            if digest in exact_hashes:
                counters["rejected_exact_duplicate"] += 1
                if exact_hashes[digest][1] != body_type:
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
            exact_hashes[digest] = (member_name, body_type)
            near_index.add(metrics["dhash"], member_name, split)
            mapped_class_names[body_type].add(class_name)
            sample_key = hashlib.sha1(member_name.encode("utf-8")).hexdigest()[:16]
            target_dir = output_root / "crops" / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"stanford_{sample_key}.jpg"
            image.save(target_path, format="JPEG", quality=95, optimize=True)
            rows.append(
                {
                    "image_path": target_path.relative_to(output_root).as_posix(),
                    "body_type": body_type,
                    "color": "unknown",
                    "crop_quality": metrics["crop_quality"],
                    "viewpoint": "unknown",
                    "blur": str(metrics["blur"]).lower(),
                    "occluded": "false",
                    "truncated": "false",
                    "night": str(metrics["night"]).lower(),
                    "camera_id": f"stanford_unknown_camera_{sample_key}",
                    "video_id": f"stanford_unknown_video_{sample_key}",
                    "track_group": f"stanford_image_{sample_key}",
                    "split": split,
                    "source_frame_id": member_name,
                    "review_status": "approved",
                    "body_type_supervised": "true",
                    "color_supervised": "false",
                    "source_dataset": "Stanford-Cars-196",
                    "source_class_name": class_name,
                    "source_license": "research-benchmark; original-dataset-terms",
                    "review_method": "official_explicit_body_keyword+decode+quality+dhash_dedup",
                    "sha256": digest,
                    "dhash64": f"{metrics['dhash']:016x}",
                    "width": str(metrics["width"]),
                    "height": str(metrics["height"]),
                    "gray_mean": str(metrics["gray_mean"]),
                    "gray_stddev": str(metrics["gray_stddev"]),
                    "edge_variance": str(metrics["edge_variance"]),
                }
            )
            counters[f"accepted_{split}_{body_type}"] += 1

    split_body_counts = Counter(f"{row['split']}:{row['body_type']}" for row in rows)
    for split in ("validation", "test"):
        missing = sorted(body for body in {"sedan", "suv", "mpv"} if not split_body_counts[f"{split}:{body}"])
        if missing:
            raise RuntimeError(f"{split} is missing mapped body types: {missing}")
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
            "name": "Stanford Cars 196",
            "official_homepage": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
            "mirror": "https://www.kaggle.com/datasets/cyizhuo/stanford-cars-by-classes-folder",
            "archive_sha256": sha256_file(archive_path),
            "license": "research benchmark; original site does not state an SPDX license",
            "license_review_ref": args.license_review_ref,
        },
        "mapping_policy": "parse mirror directories that preserve official class names; accept only explicit Sedan, SUV, or Minivan keywords",
        "mapped_class_names": {key: sorted(value) for key, value in mapped_class_names.items()},
        "excluded_policy": "all ambiguous model-only, coupe, convertible, hatchback, wagon, and sports classes",
        "color_policy": "unknown with color_supervised=false",
        "split_policy": "official test preserved; class-stratified validation carved from official train; global perceptual dedup protects evaluation",
        "unmatched_image_members": unmatched_images,
        "rows": len(rows),
        "split_body_counts": dict(split_body_counts),
        "audit_counters": dict(counters),
    }
    with (output_root / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"PASS: wrote {len(rows)} audited Stanford Cars samples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
