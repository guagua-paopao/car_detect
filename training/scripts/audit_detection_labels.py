#!/usr/bin/env python3
"""Copy a detection dataset and add only high-confidence missed vehicle boxes."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, write_json  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEACHER_TO_VCAS = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--teacher", default="yolo11x.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--conflict-iou", type=float, default=0.70)
    parser.add_argument("--minimum-area", type=float, default=0.0002)
    parser.add_argument("--device", default="0")
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


def iou(
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


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detection label audit") from exc

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    labels = load_json(args.labels)
    class_names = [str(value) for value in labels["vehicle_classes"]]
    class_index = {name: index for index, name in enumerate(class_names)}
    missing_contract_labels = set(TEACHER_TO_VCAS.values()) - set(class_index)
    if missing_contract_labels:
        raise RuntimeError(
            f"local detection label contract is missing {missing_contract_labels}"
        )
    copy_dataset(source_root, output_root)
    write_yaml(output_root, class_names)

    model = YOLO(args.teacher)
    image_root = output_root / "images" / "train"
    results = model.predict(
        source=str(image_root),
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.confidence,
        iou=0.60,
        device=args.device,
        verbose=False,
    )
    counters: Counter[str] = Counter()
    added_by_class: Counter[str] = Counter()
    conflicts: list[dict[str, Any]] = []
    for result in results:
        image_path = Path(result.path)
        label_path = output_root / "labels" / "train" / f"{image_path.stem}.txt"
        lines = [
            line
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing = [yolo_to_xyxy(line.split()) for line in lines]
        additions: list[str] = []
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxyn.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        names = result.names
        for box_values, teacher_class_id, confidence in zip(
            boxes, class_ids, confidences
        ):
            teacher_name = str(names[int(teacher_class_id)]).lower()
            mapped = TEACHER_TO_VCAS.get(teacher_name)
            if mapped is None:
                continue
            box = tuple(max(0.0, min(1.0, float(value))) for value in box_values)
            if (box[2] - box[0]) * (box[3] - box[1]) < args.minimum_area:
                counters["rejected_small"] += 1
                continue
            local_class_id = class_index[mapped]
            same_iou = max(
                (
                    iou(box, existing_box)
                    for existing_class, existing_box in existing
                    if existing_class == local_class_id
                ),
                default=0.0,
            )
            if same_iou >= args.match_iou:
                counters["matched_existing"] += 1
                continue
            conflicting = [
                (existing_class, iou(box, existing_box))
                for existing_class, existing_box in existing
                if existing_class != local_class_id
                and iou(box, existing_box) >= args.conflict_iou
            ]
            if conflicting:
                counters["rejected_class_conflict"] += 1
                if len(conflicts) < 500:
                    conflicts.append(
                        {
                            "image_path": str(image_path),
                            "teacher_class": mapped,
                            "teacher_confidence": float(confidence),
                            "conflicts": conflicting,
                        }
                    )
                continue
            additions.append(xyxy_to_yolo(local_class_id, box))
            existing.append((local_class_id, box))
            counters["added"] += 1
            added_by_class[mapped] += 1
        if additions:
            label_path.write_text(
                "\n".join([*lines, *additions]) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            counters["images_changed"] += 1
        counters["images_processed"] += 1

    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "source_root": str(source_root),
        "output_root": str(output_root),
        "teacher": args.teacher,
        "policy": {
            "train_only": True,
            "preserve_existing_labels": True,
            "confidence": args.confidence,
            "batch_size": args.batch_size,
            "match_iou": args.match_iou,
            "conflict_iou": args.conflict_iou,
            "minimum_area": args.minimum_area,
        },
        "counts": dict(counters),
        "added_by_class": dict(added_by_class),
        "class_names": class_names,
        "conflicts": conflicts,
    }
    write_json(args.report.resolve(), report)
    print(f"PASS: detection label audit complete: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
