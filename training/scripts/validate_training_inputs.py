#!/usr/bin/env python3
"""Validate YOLO detection data and/or the VCAS attribute CSV before training."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, parse_bool, write_json  # noqa: E402
from src.splits import find_group_leakage  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ATTRIBUTE_FIELDS = {
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
    "review_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-root", type=Path)
    parser.add_argument("--attribute-csv", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--min-detection-images", type=int, default=0)
    parser.add_argument("--min-attribute-crops", type=int, default=0)
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="apply the minimum to approved, non-poor, supervised crops",
    )
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def validate_detection(
    root: Path,
    minimum: int,
    vehicle_classes: list[str],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    seen_hashable_paths: set[str] = set()
    total_boxes = 0
    for split in ("train", "validation", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing split directories for {split}")
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image_path.relative_to(root).as_posix()
            if relative in seen_hashable_paths:
                errors.append(f"duplicate image path: {relative}")
            seen_hashable_paths.add(relative)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"unreadable image {image_path}: {exc}")
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"missing label file for {image_path}")
                continue
            split_counts[split] += 1
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
            if not lines:
                errors.append(f"empty label file: {label_path}")
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = (
                        float(value) for value in parts[1:]
                    )
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric YOLO label")
                    continue
                maximum_class_id = len(vehicle_classes) - 1
                if class_id not in range(len(vehicle_classes)):
                    errors.append(
                        f"{label_path}:{line_number}: class id must be "
                        f"0..{maximum_class_id}"
                    )
                if not (
                    0 < width <= 1
                    and 0 < height <= 1
                    and 0 <= center_x <= 1
                    and 0 <= center_y <= 1
                    and center_x - width / 2 >= -1e-6
                    and center_y - height / 2 >= -1e-6
                    and center_x + width / 2 <= 1 + 1e-6
                    and center_y + height / 2 <= 1 + 1e-6
                ):
                    errors.append(f"{label_path}:{line_number}: invalid normalized box")
                class_counts[str(class_id)] += 1
                total_boxes += 1
    total_images = sum(split_counts.values())
    if total_images < minimum:
        errors.append(f"detection image count {total_images} is below required {minimum}")
    return (
        {
            "images": total_images,
            "boxes": total_boxes,
            "class_names": vehicle_classes,
            "split_counts": dict(split_counts),
            "class_counts": dict(class_counts),
        },
        errors,
    )


def validate_attributes(
    csv_path: Path,
    labels: dict[str, Any],
    minimum: int,
    require_approved: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}, ["attribute CSV has no header"]
        missing = ATTRIBUTE_FIELDS - set(reader.fieldnames)
        if missing:
            return {}, [f"attribute CSV is missing fields: {sorted(missing)}"]
        rows = list(reader)
    body_types = set(labels["body_types"])
    colors = set(labels["colors"])
    qualities = set(labels["crop_qualities"])
    viewpoints = set(labels["viewpoints"])
    split_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    supervised_split_counts: Counter[str] = Counter()
    supervised_head_counts: Counter[str] = Counter()
    root = csv_path.parent
    seen_paths: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        prefix = f"{csv_path}:{row_number}"
        image_relative = row["image_path"].replace("\\", "/")
        image_path = root / image_relative
        if image_relative in seen_paths:
            errors.append(f"{prefix}: duplicate image_path {image_relative}")
        seen_paths.add(image_relative)
        if not image_path.is_file():
            errors.append(f"{prefix}: missing crop {image_path}")
        review_status = row["review_status"].strip()
        approved = review_status == "approved"
        head_supervised: dict[str, bool] = {}
        for field in ("body_type", "color"):
            column = f"{field}_supervised"
            if column in row and row[column].strip():
                try:
                    head_supervised[field] = parse_bool(row[column], column)
                except ValueError as exc:
                    errors.append(f"{prefix}: {exc}")
                    head_supervised[field] = False
            else:
                head_supervised[field] = approved
        if approved or row["body_type"].strip():
            if row["body_type"] not in body_types:
                errors.append(f"{prefix}: invalid body_type {row['body_type']!r}")
        if approved or row["color"].strip():
            if row["color"] not in colors:
                errors.append(f"{prefix}: invalid color {row['color']!r}")
        if approved or row["crop_quality"].strip():
            if row["crop_quality"] not in qualities:
                errors.append(f"{prefix}: invalid crop_quality {row['crop_quality']!r}")
        if approved or row["viewpoint"].strip():
            if row["viewpoint"] not in viewpoints:
                errors.append(f"{prefix}: invalid viewpoint {row['viewpoint']!r}")
        if row["split"] not in {"train", "validation", "test"}:
            errors.append(f"{prefix}: invalid split {row['split']!r}")
        for field in ("blur", "occluded", "truncated", "night"):
            if approved or row[field].strip():
                try:
                    parse_bool(row[field], field)
                except ValueError as exc:
                    errors.append(f"{prefix}: {exc}")
        if approved and row["crop_quality"] == "poor" and (
            row["body_type"] != "unknown" or row["color"] != "unknown"
        ):
            errors.append(f"{prefix}: poor crop must use unknown for both attributes")
        if approved and row["crop_quality"] == "poor" and not all(
            head_supervised.values()
        ):
            errors.append(
                f"{prefix}: poor crop must explicitly supervise unknown for both heads"
            )
        for field in ("body_type", "color"):
            if approved and not head_supervised[field] and row[field] != "unknown":
                errors.append(
                    f"{prefix}: unsupervised {field} head must store unknown"
                )
            if approved and head_supervised[field]:
                supervised_head_counts[field] += 1
        split_counts[row["split"]] += 1
        body_counts[row["body_type"]] += 1
        color_counts[row["color"]] += 1
        quality_counts[row["crop_quality"]] += 1
        review_counts[review_status] += 1
        if (
            approved
            and row["crop_quality"] != "poor"
            and (
                (
                    head_supervised["body_type"]
                    and row["body_type"] in body_types - {"unknown"}
                )
                or (
                    head_supervised["color"]
                    and row["color"] in colors - {"unknown"}
                )
            )
        ):
            supervised_split_counts[row["split"]] += 1

    errors.extend(find_group_leakage(rows))
    supervised_crops = sum(supervised_split_counts.values())
    gate_count = supervised_crops if require_approved else len(rows)
    if gate_count < minimum:
        gate_name = "approved supervised attribute crop count" if require_approved else "attribute crop count"
        errors.append(f"{gate_name} {gate_count} is below required {minimum}")
    return (
        {
            "crops": len(rows),
            "approved_supervised_crops": supervised_crops,
            "split_counts": dict(split_counts),
            "supervised_split_counts": dict(supervised_split_counts),
            "supervised_head_counts": dict(supervised_head_counts),
            "body_type_counts": dict(body_counts),
            "color_counts": dict(color_counts),
            "quality_counts": dict(quality_counts),
            "review_counts": dict(review_counts),
        },
        errors,
    )


def main() -> int:
    args = parse_args()
    if args.detection_root is None and args.attribute_csv is None:
        raise SystemExit("provide --detection-root and/or --attribute-csv")
    labels = load_json(args.labels)
    summary: dict[str, Any] = {"status": "pass"}
    errors: list[str] = []
    if args.detection_root is not None:
        result, found = validate_detection(
            args.detection_root.resolve(),
            args.min_detection_images,
            [str(value) for value in labels["vehicle_classes"]],
        )
        summary["detection"] = result
        errors.extend(found)
    if args.attribute_csv is not None:
        result, found = validate_attributes(
            args.attribute_csv.resolve(),
            labels,
            args.min_attribute_crops,
            args.require_approved,
        )
        summary["attributes"] = result
        errors.extend(found)
    if errors:
        summary["status"] = "fail"
        summary["errors"] = errors
    if args.summary:
        write_json(args.summary, summary)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"PASS: training inputs validated: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
