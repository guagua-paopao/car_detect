#!/usr/bin/env python3
"""Build an R3 detector set with mutually exclusive vehicle labels.

Open Images ``Vehicle`` and ``Land vehicle`` boxes are hierarchical parent
labels.  A flat YOLO head cannot learn them alongside car/bus/truck/motorcycle
without producing cross-class duplicates.  This builder uses an independent
COCO teacher only to refine parent boxes into one of the four concrete project
classes.  Parent boxes that cannot be refined confidently are excluded instead
of being used as contradictory supervision.  The six-class output contract is
kept unchanged; ``vehicle`` remains reserved for runtime fallback semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]
COCO_TO_PROJECT = {2: 0, 3: 3, 5: 1, 7: 2}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--teacher-chunk",
        type=int,
        default=256,
        help="Maximum images submitted to Ultralytics per predict call.",
    )
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.55)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    parser.add_argument("--conflict-iou", type=float, default=0.75)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def xywh_to_xyxy(values: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = values
    return (x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0)


def xyxy_to_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    return f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}"


def read_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    labels: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        box = xywh_to_xyxy(float(value) for value in fields[1:])
        # YOLO text rounding can place an edge a few nanounits outside [0, 1].
        if not (-1e-6 <= box[0] < box[2] <= 1.0 + 1e-6 and -1e-6 <= box[1] < box[3] <= 1.0 + 1e-6):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {box}")
        box = tuple(max(0.0, min(1.0, value)) for value in box)
        labels.append((class_id, box))
    return labels


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)

    from ultralytics import YOLO

    inventory: dict[str, list[dict[str, object]]] = {}
    parent_images: list[Path] = []
    counters: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        label_dir = source_root / "labels" / split
        image_dir = source_root / "images" / split
        images_by_stem: dict[str, Path] = {}
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image_path.stem in images_by_stem:
                raise RuntimeError(f"duplicate image stem in {image_dir}: {image_path.stem}")
            images_by_stem[image_path.stem] = image_path
        records: list[dict[str, object]] = []
        for label_path in sorted(label_dir.glob("*.txt")):
            labels = read_labels(label_path)
            image_path = images_by_stem.get(label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"missing image for {label_path}")
            record = {"label": label_path, "image": image_path, "labels": labels}
            records.append(record)
            counters[f"source_{split}_images"] += 1
            for class_id, _ in labels:
                counters[f"source_{split}_box_{class_id}"] += 1
            if any(class_id == 4 for class_id, _ in labels):
                parent_images.append(image_path)
        inventory[split] = records

    print(
        f"inventory complete: {sum(len(records) for records in inventory.values())} images, "
        f"{len(parent_images)} images require parent-label refinement",
        flush=True,
    )

    teacher = YOLO(str(args.teacher.resolve()))
    prediction_map: dict[Path, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    total_parent_images = len(parent_images)
    for chunk_start in range(0, total_parent_images, args.teacher_chunk):
        chunk_paths = parent_images[chunk_start:chunk_start + args.teacher_chunk]
        results = teacher.predict(
            source=[str(path) for path in chunk_paths],
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.confidence,
            iou=0.70,
            classes=sorted(COCO_TO_PROJECT),
            device=args.device,
            half=True,
            verbose=False,
        )
        for image_path, result in zip(chunk_paths, results, strict=True):
            predictions: list[tuple[int, float, tuple[float, float, float, float]]] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxyn.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for raw_class, confidence, raw_box in zip(classes, confidences, boxes, strict=True):
                    coco_class = int(raw_class)
                    mapped = COCO_TO_PROJECT.get(coco_class)
                    if mapped is None:
                        continue
                    predictions.append((mapped, float(confidence), tuple(float(v) for v in raw_box)))
            prediction_map[image_path] = predictions
        completed = min(chunk_start + len(chunk_paths), total_parent_images)
        print(f"teacher refinement: {completed}/{total_parent_images} images", flush=True)

    output_root.mkdir(parents=True)
    for split, records in inventory.items():
        for record in records:
            image_path = record["image"]
            labels = record["labels"]
            assert isinstance(image_path, Path)
            assert isinstance(labels, list)
            predictions = prediction_map.get(image_path, [])
            candidates: list[tuple[int, tuple[float, float, float, float], str]] = []
            for class_id, box in labels:
                if class_id != 4:
                    candidates.append((class_id, box, "human"))
                    continue
                best = None
                for mapped_class, confidence, predicted_box in predictions:
                    overlap = iou(box, predicted_box)
                    score = overlap * confidence
                    if overlap >= args.match_iou and (best is None or score > best[0]):
                        best = (score, mapped_class, confidence, predicted_box, overlap)
                if best is None:
                    counters[f"dropped_{split}_unresolved_vehicle"] += 1
                    continue
                _, mapped_class, confidence, predicted_box, overlap = best
                # Preserve the official human box geometry and refine only its class.
                candidates.append((mapped_class, box, "teacher_refined"))
                counters[f"relabeled_{split}_vehicle_to_{mapped_class}"] += 1
                counters[f"teacher_match_conf_sum_{split}"] += int(confidence * 1_000_000)
                counters[f"teacher_match_iou_sum_{split}"] += int(overlap * 1_000_000)

            accepted: list[tuple[int, tuple[float, float, float, float], str]] = []
            for candidate in sorted(candidates, key=lambda item: 0 if item[2] == "human" else 1):
                class_id, box, source = candidate
                if any(existing_class == class_id and iou(box, existing_box) >= args.duplicate_iou
                       for existing_class, existing_box, _ in accepted):
                    counters[f"skipped_{split}_duplicate"] += 1
                    continue
                if source == "teacher_refined" and any(
                    existing_class != class_id and iou(box, existing_box) >= args.conflict_iou
                    for existing_class, existing_box, _ in accepted
                ):
                    counters[f"skipped_{split}_teacher_conflict"] += 1
                    continue
                accepted.append(candidate)

            if not accepted:
                counters[f"dropped_{split}_empty_images"] += 1
                continue
            destination_image = output_root / "images" / split / image_path.name
            destination_label = output_root / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy(image_path, destination_image)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text(
                "\n".join(xyxy_to_line(class_id, box) for class_id, box, _ in accepted) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            counters[f"output_{split}_images"] += 1
            for class_id, _, _ in accepted:
                counters[f"output_{split}_box_{class_id}"] += 1

    yaml_path = output_root / "vehicle_det_v1.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "dataset_version": "dataset-large-v1-sixclass-ontology-r5",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "teacher": str(args.teacher.resolve()),
        "class_names": CLASS_NAMES,
        "policy": {
            "independent_coco_teacher": True,
            "flat_head_parent_labels_removed": True,
            "vehicle_reserved_for_runtime_fallback": True,
            "confidence": args.confidence,
            "match_iou": args.match_iou,
            "duplicate_iou": args.duplicate_iou,
            "conflict_iou": args.conflict_iou,
            "seed": args.seed,
        },
        "counts": dict(sorted(counters.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
