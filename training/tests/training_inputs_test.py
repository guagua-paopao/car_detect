#!/usr/bin/env python3
"""Dependency-light contract tests for the VCAS training input tools."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

from src.common import load_json, trainable_attribute_labels  # noqa: E402
from src.splits import assign_group_splits, find_group_leakage  # noqa: E402
from validate_training_inputs import (  # noqa: E402
    validate_attributes,
    validate_detection,
)


ATTRIBUTE_FIELDS = [
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
]


def write_attribute_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTRIBUTE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    assigned_a = assign_group_splits([f"group-{index}" for index in range(20)])
    assigned_b = assign_group_splits([f"group-{index}" for index in range(20)])
    if assigned_a != assigned_b:
        raise AssertionError("group split must be deterministic")

    leaked = [
        {
            "camera_id": "camera-a",
            "video_id": "video-a",
            "track_group": "track-a",
            "split": "train",
        },
        {
            "camera_id": "camera-a",
            "video_id": "video-b",
            "track_group": "track-b",
            "split": "test",
        },
    ]
    if not find_group_leakage(leaked):
        raise AssertionError("group leakage must be detected")

    labels = load_json(ROOT / "config" / "vehicle_labels.v1.json")
    body_types, colors = trainable_attribute_labels(labels)
    if body_types != labels["body_types"] or colors != labels["colors"]:
        raise AssertionError(
            "attribute training outputs must preserve the full canonical order, "
            "including unknown"
        )
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        detection_root = temp / "detection"
        for split in ("train", "validation", "test"):
            image_dir = detection_root / "images" / split
            label_dir = detection_root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image_path = image_dir / f"{split}-001.jpg"
            Image.new("RGB", (64, 64), color=(128, 128, 128)).save(image_path)
            (label_dir / f"{split}-001.txt").write_text(
                "0 0.500000 0.500000 0.500000 0.500000\n",
                encoding="utf-8",
            )
        summary, errors = validate_detection(
            detection_root, 3, list(labels["vehicle_classes"])
        )
        if errors or summary["images"] != 3 or summary["boxes"] != 3:
            raise AssertionError(f"valid detection fixture failed: {summary}, {errors}")

        invalid_label = detection_root / "labels" / "train" / "train-001.txt"
        invalid_label.write_text("9 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        _, errors = validate_detection(
            detection_root, 3, list(labels["vehicle_classes"])
        )
        if not any("class id must be 0..5" in error for error in errors):
            raise AssertionError(f"invalid class id was not rejected: {errors}")

        attribute_root = temp / "attributes"
        crops = attribute_root / "crops"
        crops.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        for index, split in enumerate(("train", "validation", "test")):
            crop_name = f"crop-{index}.jpg"
            Image.new("RGB", (96, 64), color=(200, 200, 200)).save(crops / crop_name)
            rows.append(
                {
                    "image_path": f"crops/{crop_name}",
                    "body_type": "suv",
                    "color": "white",
                    "crop_quality": "good",
                    "viewpoint": "side",
                    "blur": "false",
                    "occluded": "false",
                    "truncated": "false",
                    "night": "false",
                    "camera_id": f"camera-{index}",
                    "video_id": f"video-{index}",
                    "track_group": f"track-{index}",
                    "split": split,
                    "review_status": "approved",
                }
            )
        attribute_csv = attribute_root / "attribute_manifest.csv"
        write_attribute_csv(attribute_csv, rows)
        summary, errors = validate_attributes(attribute_csv, labels, 3)
        if errors or summary["crops"] != 3:
            raise AssertionError(f"valid attribute fixture failed: {summary}, {errors}")
        if summary["approved_supervised_crops"] != 3:
            raise AssertionError(f"approved supervised count is wrong: {summary}")

        rows[2]["review_status"] = "pending"
        write_attribute_csv(attribute_csv, rows)
        summary, errors = validate_attributes(
            attribute_csv,
            labels,
            3,
            require_approved=True,
        )
        if not any(
            "approved supervised attribute crop count 2 is below required 3" in error
            for error in errors
        ):
            raise AssertionError(
                f"pending crops were incorrectly accepted by the formal gate: "
                f"{summary}, {errors}"
            )
        rows[2]["review_status"] = "approved"

        rows[2]["track_group"] = rows[0]["track_group"]
        write_attribute_csv(attribute_csv, rows)
        _, errors = validate_attributes(attribute_csv, labels, 3)
        if not any("track_group" in error and "appears in" in error for error in errors):
            raise AssertionError(f"attribute leakage was not rejected: {errors}")

        rows[2]["track_group"] = "track-2"
        rows[0]["crop_quality"] = "poor"
        write_attribute_csv(attribute_csv, rows)
        _, errors = validate_attributes(attribute_csv, labels, 3)
        if not any("poor crop must use unknown" in error for error in errors):
            raise AssertionError(f"invalid poor crop was not rejected: {errors}")

    print("PASS: VCAS training input, split, leakage, and label checks")


if __name__ == "__main__":
    main()
