#!/usr/bin/env python3
"""Calibrate a fixed detector confidence threshold across named validation domains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NAME", "IMAGE_DIR", "LABEL_DIR"),
        required=True,
    )
    parser.add_argument("--thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--minimum-confidence", type=float, default=0.10)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-domain-recall", type=float, default=0.70)
    parser.add_argument("--min-bmd-recall", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def load_ground_truth(path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    rows: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, box_width, box_height = line.split()
        center_x = float(cx) * width
        center_y = float(cy) * height
        w = float(box_width) * width
        h = float(box_height) * height
        rows.append(
            (
                int(class_id),
                (center_x - w / 2.0, center_y - h / 2.0, center_x + w / 2.0, center_y + h / 2.0),
            )
        )
    return rows


def match_image(
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    threshold: float,
    match_iou: float,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    classes = sorted({row[0] for row in ground_truth} | {row[0] for row in predictions})
    for class_id in classes:
        truth = [box for current, box in ground_truth if current == class_id]
        predicted = sorted(
            ((confidence, box) for current, confidence, box in predictions if current == class_id and confidence >= threshold),
            reverse=True,
        )
        used: set[int] = set()
        for _, box in predicted:
            best: tuple[float, int] | None = None
            for index, truth_box in enumerate(truth):
                if index in used:
                    continue
                overlap = iou(box, truth_box)
                if overlap >= match_iou and (best is None or overlap > best[0]):
                    best = (overlap, index)
            if best is None:
                counts["fp"] += 1
                counts[f"class_{class_id}_fp"] += 1
            else:
                used.add(best[1])
                counts["tp"] += 1
                counts[f"class_{class_id}_tp"] += 1
        missed = len(truth) - len(used)
        counts["fn"] += missed
        counts[f"class_{class_id}_fn"] += missed
    return counts


def metrics(counts: Counter[str]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    if min(thresholds) < args.minimum_confidence:
        raise ValueError("minimum-confidence must not exceed the smallest threshold")
    model = YOLO(str(args.model.resolve()))
    domain_results: dict[str, dict[str, Any]] = {}
    for raw_name, raw_images, raw_labels in args.dataset:
        image_dir = Path(raw_images).resolve()
        label_dir = Path(raw_labels).resolve()
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        counters = {threshold: Counter() for threshold in thresholds}
        processed = 0
        for offset in range(0, len(images), args.batch):
            batch_paths = images[offset : offset + args.batch]
            results = model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                conf=args.minimum_confidence,
                iou=0.70,
                batch=args.batch,
                device=args.device,
                half=True,
                verbose=False,
            )
            for image_path, result in zip(batch_paths, results, strict=True):
                height, width = result.orig_shape
                truth = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
                predictions: list[tuple[int, float, tuple[float, float, float, float]]] = []
                if result.boxes is not None:
                    for class_id, confidence, box in zip(
                        result.boxes.cls.detach().cpu().tolist(),
                        result.boxes.conf.detach().cpu().tolist(),
                        result.boxes.xyxy.detach().cpu().tolist(),
                        strict=True,
                    ):
                        predictions.append(
                            (int(class_id), float(confidence), tuple(float(value) for value in box))
                        )
                for threshold in thresholds:
                    counters[threshold].update(
                        match_image(truth, predictions, threshold, args.match_iou)
                    )
                processed += 1
            if processed % 1000 < args.batch:
                print(f"{raw_name}: {processed}/{len(images)}", flush=True)
        domain_results[raw_name] = {
            "images": len(images),
            "thresholds": {
                f"{threshold:.2f}": metrics(counters[threshold]) for threshold in thresholds
            },
        }

    candidates: list[tuple[float, float]] = []
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        rows = [domain_results[name]["thresholds"][key] for name in domain_results]
        original = domain_results.get("original_validation", {}).get("thresholds", {}).get(key)
        bmd = domain_results.get("bmd_validation", {}).get("thresholds", {}).get(key)
        if original and original["recall"] < args.min_domain_recall:
            continue
        if bmd and bmd["recall"] < args.min_bmd_recall:
            continue
        candidates.append((min(float(row["f1"]) for row in rows), threshold))
    if not candidates:
        recommendation = None
        status = "no_threshold_passed_recall_constraints"
    else:
        recommendation = max(candidates)[1]
        status = "pass"
    payload = {
        "schema_version": "1.0",
        "model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "minimum_inference_confidence": args.minimum_confidence,
        "match_iou": args.match_iou,
        "selection": {
            "status": status,
            "recommended_confidence": recommendation,
            "objective": "maximize the minimum F1 across validation domains",
            "min_original_recall": args.min_domain_recall,
            "min_bmd_recall": args.min_bmd_recall,
        },
        "domains": domain_results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload["selection"], ensure_ascii=False, indent=2))
    return 0 if recommendation is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
