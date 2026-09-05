#!/usr/bin/env python3
"""Merge a train-only detection supplement into a versioned six-class dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402
from import_vcor_colors import NearDuplicateIndex, dhash64  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json")
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def label_path(root: Path, split: str, image_path: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def validate_and_count(path: Path, class_count: int, counts: Counter[str], prefix: str) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty label file: {path}")
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO row in {path}: {line}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if not 0 <= class_id < class_count or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"out-of-contract YOLO row in {path}: {line}")
        counts[f"{prefix}_box_{class_id}"] += 1


def write_yaml(root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/validation",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    (root / "vehicle_det_v1.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    base_root = args.base_root.resolve()
    supplement_root = args.supplement_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    labels = load_json(args.labels)
    class_names = list(labels["vehicle_classes"])
    expected = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]
    if class_names != expected:
        raise RuntimeError(f"unexpected detector contract: {class_names}")

    counts: Counter[str] = Counter()
    known_hashes: set[str] = set()
    known_stems: set[str] = set()
    near_index = NearDuplicateIndex(args.max_dhash_distance)
    # Keep held-out samples when an exact cross-split duplicate exists.
    for split in ("test", "validation", "train"):
        for image_path in sorted((base_root / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            source_label = label_path(base_root, split, image_path)
            digest = sha256_file(image_path)
            if digest in known_hashes:
                counts[f"rejected_exact_duplicate_base_{split}"] += 1
                continue
            with Image.open(image_path) as image:
                image.load()
                perceptual = dhash64(image.convert("RGB"))
            match = near_index.match(perceptual)
            if match is not None:
                counts[f"rejected_near_duplicate_base_{split}"] += 1
                if match[1] != split:
                    counts["rejected_cross_split_near_duplicate"] += 1
                continue
            validate_and_count(source_label, len(class_names), counts, f"{split}")
            known_hashes.add(digest)
            known_stems.add(image_path.stem)
            near_index.add(perceptual, image_path.name, split)
            link_or_copy(image_path, output_root / "images" / split / image_path.name)
            target_label = output_root / "labels" / split / source_label.name
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, target_label)
            counts[f"base_{split}_images"] += 1

    for image_path in sorted((supplement_root / "images" / "train").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        source_label = label_path(supplement_root, "train", image_path)
        validate_and_count(source_label, len(class_names), counts, "supplement_train")
        digest = sha256_file(image_path)
        if image_path.stem in known_stems or digest in known_hashes:
            counts["rejected_exact_duplicate_supplement"] += 1
            continue
        with Image.open(image_path) as image:
            image.load()
            perceptual = dhash64(image.convert("RGB"))
        match = near_index.match(perceptual)
        if match is not None:
            counts["rejected_near_duplicate_supplement"] += 1
            continue
        known_hashes.add(digest)
        known_stems.add(image_path.stem)
        near_index.add(perceptual, image_path.name, "train")
        link_or_copy(image_path, output_root / "images" / "train" / image_path.name)
        target_label = output_root / "labels" / "train" / source_label.name
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_label, target_label)
        counts["supplement_train_images"] += 1

    write_yaml(output_root, class_names)
    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "base_root": str(base_root),
        "supplement_root": str(supplement_root),
        "output_root": str(output_root),
        "policy": {
            "supplement_train_only": True,
            "preserve_base_validation_and_test": True,
            "reject_exact_duplicates": True,
            "reject_near_duplicates": True,
            "max_dhash_distance": args.max_dhash_distance,
        },
        "counts": dict(counts),
        "class_names": class_names,
    }
    write_json(args.report.resolve(), report)
    print(f"PASS: merged detection supplement: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
