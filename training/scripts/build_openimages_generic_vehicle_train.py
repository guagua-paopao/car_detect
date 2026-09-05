#!/usr/bin/env python3
"""Build a human-labeled Open Images supplement for the generic vehicle class."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


DIRECT_MAPPING = {
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
GENERIC_NAMES = {"Vehicle", "Land vehicle"}
EXPECTED_CLASSES = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json")
    parser.add_argument("--samples-per-query", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--minimum-area", type=float, default=0.0002)
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument("--license-approval-ref", required=True)
    return parser.parse_args()


def sample_id(sample: Any) -> str:
    value = getattr(sample, "open_images_id", None)
    return str(value) if value else Path(str(sample.filepath)).stem


def iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def yolo_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return f"{class_id} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}"


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    labels = load_json(args.labels)
    class_names = list(labels["vehicle_classes"])
    if class_names != EXPECTED_CLASSES:
        raise RuntimeError(f"unexpected detector contract: {class_names}")
    if not args.license_approval_ref.strip():
        raise ValueError("license approval reference is required")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError("FiftyOne is required") from exc

    output_root = args.output_root.resolve()
    cache_dir = args.cache_dir.resolve()
    fo.config.dataset_zoo_dir = str(cache_dir)
    existing_ids = {
        path.stem[len("oi_train_") :]
        for path in (args.exclude_root.resolve() / "images" / "train").iterdir()
        if path.stem.startswith("oi_train_")
    }
    records: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()

    for query_index, query in enumerate(("Vehicle", "Land vehicle")):
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[query],
            only_matching=False,
            max_samples=args.samples_per_query,
            shuffle=True,
            seed=args.seed + query_index * 1009,
            include_id=True,
            dataset_name=f"vcas-generic-{query_index}-{args.seed}",
            drop_existing_dataset=True,
        )
        for sample in dataset:
            oid = sample_id(sample)
            if oid in existing_ids:
                counters["skipped_existing_image"] += 1
                continue
            record = records.setdefault(
                oid,
                {"source_path": Path(str(sample.filepath)), "direct": [], "generic": []},
            )
            ground_truth = getattr(sample, "ground_truth", None)
            detections = getattr(ground_truth, "detections", []) or []
            for detection in detections:
                label = str(detection.label)
                mapped = DIRECT_MAPPING.get(label)
                is_generic = label in GENERIC_NAMES
                if mapped is None and not is_generic:
                    continue
                x, y, width, height = (float(value) for value in detection.bounding_box)
                x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
                width, height = max(0.0, min(1.0 - x, width)), max(0.0, min(1.0 - y, height))
                if width * height < args.minimum_area:
                    counters["skipped_small"] += 1
                    continue
                box = (x, y, x + width, y + height)
                target = "vehicle" if is_generic else mapped
                destination = record["generic" if is_generic else "direct"]
                if any(item[0] == target and iou(box, item[1]) >= 0.95 for item in destination):
                    counters["skipped_duplicate_box"] += 1
                    continue
                destination.append((target, box))

    attribution: list[dict[str, str]] = []
    for oid, record in sorted(records.items()):
        accepted: list[tuple[str, tuple[float, float, float, float]]] = []
        for target, box in record["direct"]:
            if any(iou(box, accepted_box) >= 0.70 for accepted_target, accepted_box in accepted if accepted_target != target):
                counters["skipped_direct_conflict"] += 1
                continue
            accepted.append((target, box))
        for target, box in record["generic"]:
            if any(iou(box, accepted_box) >= args.duplicate_iou for _, accepted_box in accepted):
                counters["skipped_generic_parent"] += 1
                continue
            accepted.append((target, box))
        if not any(target == "vehicle" for target, _ in accepted):
            counters["skipped_without_independent_generic"] += 1
            continue

        source_path: Path = record["source_path"]
        stem = f"oi_generic_train_{oid}"
        image_path = output_root / "images" / "train" / f"{stem}{source_path.suffix.lower()}"
        label_path = output_root / "labels" / "train" / f"{stem}.txt"
        link_or_copy(source_path, image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(yolo_line(class_names.index(target), box) for target, box in accepted) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        counters["images"] += 1
        for target, _ in accepted:
            counters[f"box_{target}"] += 1
        attribution.append(
            {
                "sample_id": stem,
                "open_images_id": oid,
                "sha256": sha256_file(image_path),
                "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                "approval_ref": args.license_approval_ref,
            }
        )

    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "source": "Open Images V7 official human boxes",
        "queries": ["Vehicle", "Land vehicle"],
        "policy": {
            "training_split_only": True,
            "concrete_class_precedence": True,
            "generic_only_without_concrete_overlap": True,
            "minimum_area": args.minimum_area,
            "duplicate_iou": args.duplicate_iou,
        },
        "counts": dict(counters),
    }
    write_json(output_root / "build-report.json", report)
    with (output_root / "attribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution[0]))
        writer.writeheader()
        writer.writerows(attribution)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
