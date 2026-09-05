#!/usr/bin/env python3
"""Build a class-stratified Open Images vehicle detector dataset.

The four deployable concrete classes are sampled independently so that the
dataset is not dominated by cars.  Open Images human boxes are preserved;
generic ``vehicle`` and contract ``other`` boxes are added later by
``import_openimages_six_classes.py`` from the same official metadata cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


SOURCE_TO_TARGET = {
    "Car": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Motorcycle": "motorcycle",
}
CLASS_ORDER = ["car", "bus", "truck", "motorcycle"]
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--validation-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--dataset-version", default="dataset-large-v1")
    parser.add_argument("--license-approval-ref", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--minimum-area", type=float, default=0.0002)
    return parser.parse_args()


def image_id(sample: Any) -> str:
    value = getattr(sample, "open_images_id", None)
    return str(value) if value else Path(str(sample.filepath)).stem


def attribute_bool(detection: Any, name: str) -> bool:
    attributes = getattr(detection, "attributes", None)
    if not isinstance(attributes, dict):
        return False
    value = attributes.get(name)
    if hasattr(value, "value"):
        return bool(value.value)
    return bool(value) if isinstance(value, bool) else False


def iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_yaml(root: Path) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/validation",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_ORDER))
    (root / "vehicle_det_pilot.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    if labels["vehicle_classes"][:4] != CLASS_ORDER:
        raise RuntimeError("local vehicle label order does not match the release contract")
    if not args.license_approval_ref.strip():
        raise ValueError("license approval reference is required")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError("FiftyOne is required") from exc

    output_root = args.output_root.resolve()
    detection_root = output_root / "detection"
    cache_dir = (args.cache_dir or output_root / "_openimages_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(cache_dir)
    requested = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    counters: Counter[str] = Counter()

    for split in SPLITS:
        for source_name, target_name in SOURCE_TO_TARGET.items():
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[source_name],
                only_matching=True,
                max_samples=requested[split],
                shuffle=True,
                seed=args.seed + CLASS_ORDER.index(target_name) * 1009,
                include_id=True,
                dataset_name=(
                    f"vcas-{args.dataset_version}-{split}-{target_name}-{args.seed}"
                ),
                drop_existing_dataset=True,
            )
            for sample in dataset:
                oid = image_id(sample)
                record = records[split].setdefault(
                    oid,
                    {
                        "source_path": Path(str(sample.filepath)),
                        "boxes": [],
                    },
                )
                ground_truth = getattr(sample, "ground_truth", None)
                detections = getattr(ground_truth, "detections", []) or []
                for detection in detections:
                    mapped = SOURCE_TO_TARGET.get(str(detection.label))
                    if mapped is None:
                        continue
                    x, y, width, height = (float(v) for v in detection.bounding_box)
                    x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
                    width = max(0.0, min(1.0 - x, width))
                    height = max(0.0, min(1.0 - y, height))
                    box = (x, y, x + width, y + height)
                    if width * height < args.minimum_area:
                        counters["rejected_small"] += 1
                        continue
                    class_id = CLASS_ORDER.index(mapped)
                    if any(
                        existing[0] == class_id and iou(box, existing[1]) >= 0.95
                        for existing in record["boxes"]
                    ):
                        counters["duplicate_box"] += 1
                        continue
                    record["boxes"].append(
                        (
                            class_id,
                            box,
                            attribute_bool(detection, "IsOccluded"),
                            attribute_bool(detection, "IsTruncated"),
                        )
                    )
                    counters[f"box_{mapped}"] += 1

    manifest_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, str]] = []
    for split in SPLITS:
        for oid, record in sorted(records[split].items()):
            if not record["boxes"]:
                continue
            source_path: Path = record["source_path"]
            suffix = source_path.suffix.lower() or ".jpg"
            sample_id = f"oi_{split}_{oid}"
            image_path = detection_root / "images" / split / f"{sample_id}{suffix}"
            label_path = detection_root / "labels" / split / f"{sample_id}.txt"
            link_or_copy(source_path, image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            annotations: list[dict[str, Any]] = []
            for class_id, box, occluded, truncated in record["boxes"]:
                x1, y1, x2, y2 = box
                lines.append(
                    f"{class_id} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} "
                    f"{x2-x1:.8f} {y2-y1:.8f}"
                )
                annotations.append(
                    {
                        "bbox_xyxy_norm": [round(v, 8) for v in box],
                        "vehicle_class": CLASS_ORDER[class_id],
                        "occluded": occluded,
                        "truncated": truncated,
                    }
                )
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with Image.open(image_path) as image:
                width, height = image.size
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "relative_path": image_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(image_path),
                    "source_id": "OPEN-IMAGES-V7",
                    "split": split,
                    "task": "detection",
                    "group": {
                        "camera_id": f"oi_camera_{oid}",
                        "video_id": f"oi_image_{oid}",
                        "track_group": f"oi_image_{oid}",
                    },
                    "image": {"width": width, "height": height},
                    "annotations": {"detections": annotations},
                }
            )
            attribution_rows.append(
                {
                    "sample_id": sample_id,
                    "open_images_id": oid,
                    "split": split,
                    "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                    "license_url": "https://github.com/openimages/dataset/blob/main/READMEV3.md",
                    "approval_ref": args.license_approval_ref,
                }
            )
            counters[f"image_{split}"] += 1

    manifest = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "labels_version": labels["labels_version"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "detection",
        "license_review_status": "approved",
        "split_policy": {
            "source_splits": "Open Images V7 official",
            "stratified_by": CLASS_ORDER,
            "seed": args.seed,
        },
        "sources": [
            {
                "source_id": "OPEN-IMAGES-V7",
                "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                "license_url": "https://github.com/openimages/dataset/blob/main/READMEV3.md",
                "authorization_ref": args.license_approval_ref,
            }
        ],
        "samples": manifest_rows,
        "counts": dict(counters),
    }
    write_json(output_root / "dataset-large-v1.json", manifest)
    write_json(output_root / "build-report.json", {"counts": dict(counters)})
    write_yaml(detection_root)
    with (output_root / "openimages_attribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)
    print(json.dumps({"status": "pass", "counts": dict(counters)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
