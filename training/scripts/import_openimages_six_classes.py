#!/usr/bin/env python3
"""Build the six-class detector set from cached Open Images human boxes."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, write_json  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIRECT_NAME_MAPPING = {
    "Car": "car",
    "Limousine": "car",
    "Van": "car",
    "Taxi": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Motorcycle": "motorcycle",
    "Ambulance": "other",
    "Cart": "other",
    "Bicycle": "other",
    "Snowmobile": "other",
    "Golf cart": "other",
    "Segway": "other",
    "Tank": "other",
    "Train": "other",
    "Unicycle": "other",
    "Wheelchair": "other",
}
GENERIC_NAME_MAPPING = {
    "Vehicle": "vehicle",
    "Land vehicle": "vehicle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--openimages-cache-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument("--conflict-iou", type=float, default=0.70)
    parser.add_argument("--minimum-area", type=float, default=0.0002)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_dataset(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite output dataset: {output_root}"
        )
    for split in ("train", "validation", "test"):
        source_images = source_root / "images" / split
        source_labels = source_root / "labels" / split
        target_images = output_root / "images" / split
        target_labels = output_root / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_images.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                link_or_copy(image_path, target_images / image_path.name)
        for label_path in sorted(source_labels.glob("*.txt")):
            shutil.copy2(label_path, target_labels / label_path.name)


def yolo_to_xyxy(parts: list[str]) -> tuple[int, tuple[float, float, float, float]]:
    class_id = int(parts[0])
    center_x, center_y, width, height = (float(value) for value in parts[1:])
    return (
        class_id,
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
    )


def xyxy_to_yolo(
    class_id: int,
    box: tuple[float, float, float, float],
) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} "
        f"{x2 - x1:.8f} {y2 - y1:.8f}"
    )


def overlap_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


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
    (root / "vehicle_det_v1.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_class_ids(classes_path: Path) -> dict[str, str]:
    with classes_path.open("r", encoding="utf-8", newline="") as handle:
        return {display_name: label_id for label_id, display_name in csv.reader(handle)}


def source_image_id(path: Path, split: str) -> str:
    prefix = f"oi_{split}_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"unexpected source image name: {path.name}")
    return path.stem[len(prefix) :]


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    cache_root = args.openimages_cache_root.resolve()
    labels = load_json(args.labels)
    class_names = [str(value) for value in labels["vehicle_classes"]]
    expected = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]
    if class_names != expected:
        raise RuntimeError(
            f"local six-class order changed: expected {expected}, got {class_names}"
        )
    class_index = {name: index for index, name in enumerate(class_names)}
    copy_dataset(source_root, output_root)
    write_yaml(output_root, class_names)

    report_counts: Counter[str] = Counter()
    added_by_split_class: dict[str, Counter[str]] = defaultdict(Counter)
    conflicts: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        metadata_root = cache_root / split
        display_to_id = read_class_ids(metadata_root / "metadata" / "classes.csv")
        id_to_display = {label_id: name for name, label_id in display_to_id.items()}
        wanted_ids = {
            display_to_id[name]: mapped
            for name, mapped in {
                **DIRECT_NAME_MAPPING,
                **GENERIC_NAME_MAPPING,
            }.items()
            if name in display_to_id
        }
        image_dir = output_root / "images" / split
        image_by_id = {
            source_image_id(path, split): path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        candidates: dict[
            str,
            list[tuple[str, str, tuple[float, float, float, float]]],
        ] = defaultdict(list)
        detections_csv = metadata_root / "labels" / "detections.csv"
        with detections_csv.open(
            "r", encoding="utf-8", newline="", buffering=1024 * 1024
        ) as handle:
            for row in csv.DictReader(handle):
                image_id = row["ImageID"]
                mapped = wanted_ids.get(row["LabelName"])
                if image_id not in image_by_id or mapped is None:
                    continue
                if (
                    row.get("IsGroupOf") == "1"
                    or row.get("IsDepiction") == "1"
                    or row.get("IsInside") == "1"
                ):
                    report_counts["skipped_non_instance"] += 1
                    continue
                box = (
                    float(row["XMin"]),
                    float(row["YMin"]),
                    float(row["XMax"]),
                    float(row["YMax"]),
                )
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < args.minimum_area:
                    report_counts["skipped_small"] += 1
                    continue
                display_name = id_to_display[row["LabelName"]]
                kind = (
                    "generic"
                    if display_name in GENERIC_NAME_MAPPING
                    else "direct"
                )
                candidates[image_id].append((kind, mapped, box))

        for image_id, image_path in image_by_id.items():
            label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
            lines = [
                line
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            existing = [yolo_to_xyxy(line.split()) for line in lines]
            additions: list[str] = []
            ordered = sorted(
                candidates.get(image_id, []),
                key=lambda value: value[0] == "generic",
            )
            for kind, mapped, box in ordered:
                target_class = class_index[mapped]
                if any(
                    existing_class == target_class
                    and overlap_iou(box, existing_box) >= args.duplicate_iou
                    for existing_class, existing_box in existing
                ):
                    report_counts["skipped_duplicate"] += 1
                    continue
                overlaps = [
                    (existing_class, overlap_iou(box, existing_box))
                    for existing_class, existing_box in existing
                    if overlap_iou(box, existing_box) >= (
                        args.duplicate_iou
                        if kind == "generic"
                        else args.conflict_iou
                    )
                ]
                if overlaps:
                    report_counts[
                        "skipped_generic_parent"
                        if kind == "generic"
                        else "skipped_class_conflict"
                    ] += 1
                    if kind != "generic" and len(conflicts) < 500:
                        conflicts.append(
                            {
                                "split": split,
                                "image_id": image_id,
                                "mapped_class": mapped,
                                "box": box,
                                "overlaps": overlaps,
                            }
                        )
                    continue
                additions.append(xyxy_to_yolo(target_class, box))
                existing.append((target_class, box))
                report_counts["added"] += 1
                added_by_split_class[split][mapped] += 1
            if additions:
                label_path.write_text(
                    "\n".join([*lines, *additions]) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                report_counts["images_changed"] += 1
            report_counts[f"{split}_images"] += 1

    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "source_root": str(source_root),
        "output_root": str(output_root),
        "openimages_cache_root": str(cache_root),
        "mapping": {
            "direct": DIRECT_NAME_MAPPING,
            "generic_fallback": GENERIC_NAME_MAPPING,
        },
        "policy": {
            "human_openimages_boxes_only": True,
            "generic_only_when_no_specific_overlap": True,
            "duplicate_iou": args.duplicate_iou,
            "conflict_iou": args.conflict_iou,
            "minimum_area": args.minimum_area,
        },
        "counts": dict(report_counts),
        "added_by_split_class": {
            split: dict(counts)
            for split, counts in added_by_split_class.items()
        },
        "class_names": class_names,
        "conflicts": conflicts,
    }
    write_json(args.report.resolve(), report)
    print(f"PASS: six-class Open Images dataset built: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
