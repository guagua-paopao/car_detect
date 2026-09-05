#!/usr/bin/env python3
"""Build a diverse Open Images train-only supplement for detector class other."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


CORE_MAPPING = {"Car": "car", "Limousine": "car", "Van": "car", "Taxi": "car", "Bus": "bus", "Truck": "truck", "Motorcycle": "motorcycle"}
OTHER_NAMES = ("Ambulance", "Cart", "Bicycle", "Snowmobile", "Golf cart", "Segway", "Tank", "Train", "Unicycle", "Wheelchair")
GENERIC_MAPPING = {"Vehicle": "vehicle", "Land vehicle": "vehicle"}
EXPECTED_CLASSES = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json")
    parser.add_argument("--samples-per-query", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--minimum-area", type=float, default=0.0002)
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument("--conflict-iou", type=float, default=0.70)
    parser.add_argument("--license-approval-ref", required=True)
    return parser.parse_args()


def image_id(sample: Any) -> str:
    value = getattr(sample, "open_images_id", None)
    return str(value) if value else Path(str(sample.filepath)).stem


def source_id_from_stem(stem: str) -> str | None:
    for prefix in ("oi_train_", "oi_generic_train_", "oi_other_train_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return None


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

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError("FiftyOne is required") from exc

    output_root = args.output_root.resolve()
    cache_dir = args.cache_dir.resolve()
    fo.config.dataset_zoo_dir = str(cache_dir)
    excluded_ids: set[str] = set()
    for root in args.exclude_root:
        for image_path in (root.resolve() / "images" / "train").iterdir():
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                oid = source_id_from_stem(image_path.stem)
                if oid:
                    excluded_ids.add(oid)

    records: dict[str, Path] = {}
    counters: Counter[str] = Counter()
    for query_index, query in enumerate(OTHER_NAMES):
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
            dataset_name=f"vcas-other-{query_index}-{args.seed}",
            drop_existing_dataset=True,
        )
        for sample in dataset:
            oid = image_id(sample)
            if oid in excluded_ids:
                counters["skipped_existing_image"] += 1
                continue
            records.setdefault(oid, Path(str(sample.filepath)))

    metadata_root = cache_dir / "open-images-v7" / "train"
    with (metadata_root / "metadata" / "classes.csv").open("r", encoding="utf-8", newline="") as handle:
        display_to_id = {display: label_id for label_id, display in csv.reader(handle)}
    wanted = {display_to_id[name]: mapped for name, mapped in CORE_MAPPING.items() if name in display_to_id}
    wanted.update({display_to_id[name]: "other" for name in OTHER_NAMES if name in display_to_id})
    wanted.update({display_to_id[name]: mapped for name, mapped in GENERIC_MAPPING.items() if name in display_to_id})
    candidates: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    with (metadata_root / "labels" / "detections.csv").open("r", encoding="utf-8", newline="", buffering=1024 * 1024) as handle:
        for row in csv.DictReader(handle):
            oid = row["ImageID"]
            mapped = wanted.get(row["LabelName"])
            if oid not in records or mapped is None:
                continue
            if row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1" or row.get("IsInside") == "1":
                counters["skipped_non_instance"] += 1
                continue
            box = (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]))
            if (box[2] - box[0]) * (box[3] - box[1]) < args.minimum_area:
                counters["skipped_small"] += 1
                continue
            candidates[oid].append((mapped, box))

    attribution: list[dict[str, str]] = []
    priority = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "other": 1, "vehicle": 2}
    for oid, source_path in sorted(records.items()):
        accepted: list[tuple[str, tuple[float, float, float, float]]] = []
        for target, box in sorted(candidates.get(oid, []), key=lambda item: priority[item[0]]):
            if any(existing_target == target and iou(box, existing_box) >= args.duplicate_iou for existing_target, existing_box in accepted):
                counters["skipped_duplicate_box"] += 1
                continue
            overlaps_other_class = any(existing_target != target and iou(box, existing_box) >= args.conflict_iou for existing_target, existing_box in accepted)
            if overlaps_other_class:
                counters["skipped_parent_or_conflict"] += 1
                continue
            accepted.append((target, box))
        if not any(target == "other" for target, _ in accepted):
            counters["skipped_without_other"] += 1
            continue
        stem = f"oi_other_train_{oid}"
        image_path = output_root / "images" / "train" / f"{stem}{source_path.suffix.lower()}"
        label_path = output_root / "labels" / "train" / f"{stem}.txt"
        link_or_copy(source_path, image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(yolo_line(class_names.index(target), box) for target, box in accepted) + "\n", encoding="utf-8", newline="\n")
        counters["images"] += 1
        for target, _ in accepted:
            counters[f"box_{target}"] += 1
        attribution.append({"sample_id": stem, "open_images_id": oid, "sha256": sha256_file(image_path), "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation", "approval_ref": args.license_approval_ref})

    if not attribution:
        raise RuntimeError("no valid other-class samples were produced")
    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "source": "Open Images V7 official human boxes",
        "queries": list(OTHER_NAMES),
        "policy": {"train_only": True, "specific_before_other_before_generic": True, "minimum_area": args.minimum_area, "duplicate_iou": args.duplicate_iou, "conflict_iou": args.conflict_iou},
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
