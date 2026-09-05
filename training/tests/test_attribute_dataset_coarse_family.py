from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.attribute_dataset import VehicleAttributeDataset  # noqa: E402


class AttributeDatasetCoarseFamilyTest(unittest.TestCase):
    def test_coarse_car_row_is_retained_without_fabricated_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "car.jpg")
            manifest = root / "manifest.csv"
            fields = [
                "image_path", "split", "review_status", "body_type", "color",
                "body_type_supervised", "color_supervised", "coarse_body_family", "pseudo_label",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "image_path": "car.jpg", "split": "train", "review_status": "approved",
                    "body_type": "unknown", "color": "unknown", "body_type_supervised": "false",
                    "color_supervised": "false", "coarse_body_family": "car", "pseudo_label": "false",
                })
            dataset = VehicleAttributeDataset(
                manifest, split="train", body_types=["sedan", "unknown"], colors=["black", "unknown"],
                transform=lambda image: image, training=True, return_pseudo=True, return_body_family=True,
            )
            _, body, color, _, pseudo, family = dataset[0]
            self.assertEqual(body, -100)
            self.assertEqual(color, -100)
            self.assertFalse(pseudo)
            self.assertEqual(family, 1)

    def test_coarse_truck_row_is_retained_as_family_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "truck.jpg")
            manifest = root / "manifest.csv"
            fields = [
                "image_path", "split", "review_status", "body_type", "color",
                "body_type_supervised", "color_supervised", "coarse_body_family", "pseudo_label",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "image_path": "truck.jpg", "split": "train", "review_status": "approved",
                    "body_type": "unknown", "color": "unknown", "body_type_supervised": "false",
                    "color_supervised": "false", "coarse_body_family": "truck", "pseudo_label": "false",
                })
            dataset = VehicleAttributeDataset(
                manifest, split="train", body_types=["light_truck", "heavy_truck", "unknown"],
                colors=["black", "unknown"], transform=lambda image: image, training=True,
                return_pseudo=True, return_body_family=True,
            )
            _, body, color, _, pseudo, family = dataset[0]
            self.assertEqual((body, color, pseudo, family), (-100, -100, False, 2))

    def test_coarse_color_row_is_retained_without_fabricated_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "silver-gray.jpg")
            manifest = root / "manifest.csv"
            fields = [
                "image_path", "split", "review_status", "body_type", "color",
                "body_type_supervised", "color_supervised", "coarse_body_family",
                "coarse_color_group", "pseudo_label",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "image_path": "silver-gray.jpg", "split": "train", "review_status": "approved",
                    "body_type": "unknown", "color": "unknown", "body_type_supervised": "false",
                    "color_supervised": "false", "coarse_body_family": "",
                    "coarse_color_group": "silver_gray", "pseudo_label": "true",
                })
            dataset = VehicleAttributeDataset(
                manifest, split="train", body_types=["sedan", "unknown"],
                colors=["black", "gray", "silver", "yellow", "brown", "unknown"],
                transform=lambda image: image, training=True, return_pseudo=True,
                return_color_group=True,
            )
            _, body, color, _, pseudo, family, color_group = dataset[0]
            self.assertEqual((body, color, pseudo, family, color_group), (-100, -100, True, 0, 1))


if __name__ == "__main__":
    unittest.main()
