from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "training" / "scripts" / "prepare_bmd45_detection.py"


class PrepareBmd45DetectionTest(unittest.TestCase):
    def test_converts_and_composes_expected_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels_version": "vehicle-labels-v1",
                        "vehicle_classes": [
                            "car", "bus", "truck", "motorcycle", "vehicle", "other"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            base = root / "base"
            for split in ("train", "validation", "test"):
                (base / "images" / split).mkdir(parents=True)
                (base / "labels" / split).mkdir(parents=True)

            bmd = root / "bmd"
            categories = [
                {"id": index, "name": name}
                for index, name in enumerate(
                    [
                        "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck",
                        "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus",
                        "Tempo-traveller", "Bicycle", "Van",
                    ]
                )
            ]
            for source_split, category_id in (("BMD-45-Train", 1), ("BMD-45-Val", 11)):
                image_dir = bmd / source_split / "images_000"
                image_dir.mkdir(parents=True)
                (image_dir / "1.png").write_bytes(b"image")
                annotation = {
                    "images": [
                        {"id": 1, "file_name": "images_000/1.png", "width": 100, "height": 50}
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": category_id, "bbox": [10, 5, 40, 20]}
                    ],
                    "categories": categories,
                }
                (bmd / source_split / "_annotations.coco.json").write_text(
                    json.dumps(annotation), encoding="utf-8"
                )

            output = root / "output"
            report = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--bmd-root", str(bmd),
                    "--base-detection-root", str(base),
                    "--output-root", str(output),
                    "--labels", str(labels),
                    "--report", str(report),
                    "--license-review-ref", "TEST-LICENSE-REF",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            train_label = next((output / "labels" / "bmd_train").iterdir()).read_text()
            validation_label = next(
                (output / "labels" / "bmd_validation").iterdir()
            ).read_text()
            self.assertTrue(train_label.startswith("0 "))
            self.assertTrue(validation_label.startswith("5 "))
            self.assertIn(
                str(base.resolve()).replace("\\", "/"),
                (output / "vehicle_det_v1.yaml").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads(report.read_text(encoding="utf-8"))["splits"]["bmd_train"]["images"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
