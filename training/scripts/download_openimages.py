#!/usr/bin/env python3
"""Download a licensed Open Images vehicle pilot and export YOLO + VCAS manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


OPEN_IMAGES_TO_VCAS = {
    "Car": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Motorcycle": "motorcycle",
}
VCAS_CLASS_ORDER = ["car", "bus", "truck", "motorcycle"]
SPLIT_COUNTS = {
    "train": "train_samples",
    "validation": "validation_samples",
    "test": "test_samples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--train-samples", type=int, default=350)
    parser.add_argument("--validation-samples", type=int, default=75)
    parser.add_argument("--test-samples", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dataset-version", default="dataset-pilot-det-v1")
    parser.add_argument(
        "--license-approval-ref",
        required=True,
        help="project record proving that Open Images use was reviewed",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional FiftyOne download cache; defaults under output-root",
    )
    return parser.parse_args()


def _attribute_bool(detection: Any, name: str) -> bool:
    attributes = getattr(detection, "attributes", None)
    if isinstance(attributes, dict):
        value = attributes.get(name)
        if hasattr(value, "value"):
            return bool(value.value)
        if isinstance(value, bool):
            return value
    return False


def _image_id(sample: Any) -> str:
    value = getattr(sample, "open_images_id", None)
    if value:
        return str(value)
    return Path(str(sample.filepath)).stem


def _write_yolo_yaml(detection_root: Path) -> None:
    lines = [
        f"path: {detection_root.as_posix()}",
        "train: images/train",
        "val: images/validation",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(VCAS_CLASS_ORDER))
    (detection_root / "vehicle_det_pilot.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    missing = [name for name in VCAS_CLASS_ORDER if name not in labels["vehicle_classes"]]
    if missing:
        raise ValueError(f"canonical label config is missing detection classes: {missing}")
    if not args.license_approval_ref.strip():
        raise ValueError("--license-approval-ref cannot be empty")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError(
            "FiftyOne is required. Install training/requirements-cloud.txt first"
        ) from exc

    output_root = args.output_root.resolve()
    detection_root = output_root / "detection"
    cache_dir = (args.cache_dir or output_root / "_openimages_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(cache_dir)
    manifest_samples: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, str]] = []

    for split, count_attribute in SPLIT_COUNTS.items():
        requested = getattr(args, count_attribute)
        if requested <= 0:
            continue
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=list(OPEN_IMAGES_TO_VCAS),
            only_matching=True,
            max_samples=requested,
            shuffle=True,
            seed=args.seed,
            include_id=True,
            dataset_name=f"vcas-{args.dataset_version}-{split}-{args.seed}",
            drop_existing_dataset=True,
        )

        images_dir = detection_root / "images" / split
        labels_dir = detection_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset:
            image_id = _image_id(sample)
            sample_id = f"oi_{split}_{image_id}"
            source_path = Path(str(sample.filepath))
            suffix = source_path.suffix.lower() or ".jpg"
            target_image = images_dir / f"{sample_id}{suffix}"
            shutil.copy2(source_path, target_image)

            ground_truth = getattr(sample, "ground_truth", None)
            source_detections = getattr(ground_truth, "detections", []) or []
            manifest_detections: list[dict[str, Any]] = []
            yolo_lines: list[str] = []
            for detection in source_detections:
                mapped = OPEN_IMAGES_TO_VCAS.get(str(detection.label))
                if mapped is None:
                    continue
                x, y, width, height = (float(v) for v in detection.bounding_box)
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                width = max(0.0, min(1.0 - x, width))
                height = max(0.0, min(1.0 - y, height))
                if width <= 0 or height <= 0:
                    continue
                class_id = VCAS_CLASS_ORDER.index(mapped)
                center_x = x + width / 2.0
                center_y = y + height / 2.0
                yolo_lines.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} "
                    f"{width:.8f} {height:.8f}"
                )
                manifest_detections.append(
                    {
                        "bbox_xyxy_norm": [
                            round(x, 8),
                            round(y, 8),
                            round(x + width, 8),
                            round(y + height, 8),
                        ],
                        "vehicle_class": mapped,
                        "occluded": _attribute_bool(detection, "IsOccluded"),
                        "truncated": _attribute_bool(detection, "IsTruncated"),
                    }
                )

            if not manifest_detections:
                target_image.unlink(missing_ok=True)
                continue

            label_path = labels_dir / f"{sample_id}.txt"
            label_path.write_text(
                "\n".join(yolo_lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with Image.open(target_image) as image:
                width, height = image.size

            relative_image = target_image.relative_to(output_root).as_posix()
            manifest_samples.append(
                {
                    "sample_id": sample_id,
                    "relative_path": relative_image,
                    "sha256": sha256_file(target_image),
                    "source_id": "OPEN-IMAGES-V7",
                    "split": split,
                    "task": "detection",
                    "group": {
                        "camera_id": f"oi_camera_{image_id}",
                        "video_id": f"oi_image_{image_id}",
                        "track_group": f"oi_image_{image_id}",
                    },
                    "image": {"width": width, "height": height},
                    "annotations": {"detections": manifest_detections},
                }
            )
            attribution_rows.append(
                {
                    "sample_id": sample_id,
                    "open_images_id": image_id,
                    "split": split,
                    "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                    "license_url": (
                        "https://github.com/openimages/dataset/blob/main/READMEV3.md"
                    ),
                    "approval_ref": args.license_approval_ref,
                }
            )

    if not manifest_samples:
        raise RuntimeError("no usable Open Images samples were exported")

    manifest = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "labels_version": labels["labels_version"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "detection",
        "license_review_status": "approved",
        "split_policy": {
            "group_keys": ["camera_id", "video_id", "track_group"],
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "seed": args.seed,
            "closed_test_groups": [
                row["group"]["track_group"]
                for row in manifest_samples
                if row["split"] == "test"
            ],
        },
        "sources": [
            {
                "source_id": "OPEN-IMAGES-V7",
                "name": "Open Images V7 vehicle subset",
                "source_type": "public_dataset",
                "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                "license_url": (
                    "https://github.com/openimages/dataset/blob/main/READMEV3.md"
                ),
                "authorization_ref": args.license_approval_ref,
                "usage_status": "approved_noncommercial",
                "attribution_required": True,
                "redistribution_allowed": False,
                "notes": (
                    "Noncommercial VCAS learning pilot. Preserve Open Images IDs and "
                    "review per-image attribution before redistribution."
                ),
            }
        ],
        "samples": manifest_samples,
    }
    write_json(output_root / "dataset-pilot-det-v1.json", manifest)
    _write_yolo_yaml(detection_root)

    attribution_path = output_root / "openimages_attribution.csv"
    with attribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)

    print(
        f"PASS: exported {len(manifest_samples)} images to {detection_root}; "
        f"manifest={output_root / 'dataset-pilot-det-v1.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
