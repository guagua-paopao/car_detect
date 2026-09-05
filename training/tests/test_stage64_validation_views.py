from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "build_stage64_validation_views.py"
SPEC = importlib.util.spec_from_file_location("stage64_views", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage64ValidationViewsTest(unittest.TestCase):
    def test_keeps_exact_truth_and_demotes_merged_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels.json"
            labels.write_text(json.dumps({
                "body_types": ["sedan", "truck", "light_truck", "heavy_truck", "unknown"],
                "colors": ["gray", "silver", "white", "unknown"],
            }), encoding="utf-8")
            for name in ("a.jpg", "b.jpg"):
                (root / name).write_bytes(b"image")
            manifest = root / "source.csv"
            fields = ["image_path", "body_type", "color", "split", "body_type_supervised", "color_supervised"]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"image_path": "a.jpg", "body_type": "sedan", "color": "gray", "split": "validation", "body_type_supervised": "true", "color_supervised": "true"})
                writer.writerow({"image_path": "b.jpg", "body_type": "light_truck", "color": "silver_gray", "split": "validation", "body_type_supervised": "true", "color_supervised": "true"})
            output_dir = root / "views"
            report = root / "report.json"
            argv = ["build", "--source", f"hard={manifest}", "--labels", str(labels), "--output-dir", str(output_dir), "--report", str(report)]
            with mock.patch("sys.argv", argv):
                self.assertEqual(MODULE.main(), 0)
            with (output_dir / "hard.validation-taxonomy-v2.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["color"], "gray")
            self.assertEqual(rows[0]["color_supervised"], "true")
            self.assertEqual(rows[1]["color"], "unknown")
            self.assertEqual(rows[1]["color_supervised"], "false")
            self.assertTrue(Path(rows[0]["image_path"]).is_absolute())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["policy"]["test_accessed"])

    def test_frozen_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels.json"
            labels.write_text(json.dumps({
                "body_types": ["truck", "light_truck", "heavy_truck", "unknown"],
                "colors": ["gray", "silver", "unknown"],
            }), encoding="utf-8")
            image = root / "a.jpg"
            image.write_bytes(b"image")
            manifest = root / "source.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "body_type", "color", "split", "video_id"])
                writer.writeheader()
                writer.writerow({"image_path": "a.jpg", "body_type": "truck", "color": "gray", "split": "validation", "video_id": "vcas_rtsp_demo_60s"})
            argv = ["build", "--source", f"bad={manifest}", "--labels", str(labels), "--output-dir", str(root / "views"), "--report", str(root / "report.json")]
            with mock.patch("sys.argv", argv), self.assertRaises(RuntimeError):
                MODULE.main()


if __name__ == "__main__":
    unittest.main()
