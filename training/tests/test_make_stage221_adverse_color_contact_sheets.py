import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_stage221_adverse_color_contact_sheets.py"
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name("make_stage221_adverse_color_contact_sheets.py")
SPEC = importlib.util.spec_from_file_location("stage221", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage221ContactSheetTests(unittest.TestCase):
    def test_one_train_row_renders_and_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "a.png"
            Image.new("RGB", (32, 24), (20, 30, 40)).save(image)
            manifest = root / "manifest.csv"
            fields = [
                "image_path", "split", "color", "sha256", "stage220_scene_claim",
                "stage220_lowlight_score", "stage220_pixel_mean_luma",
                "stage220_frame_lowlight_proxy", "stage220_frame_night_candidate",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "image_path": str(image), "split": "train", "color": "blue",
                    "sha256": "a" * 64, "stage220_scene_claim": "night_metadata_positive",
                    "stage220_lowlight_score": "0.8", "stage220_pixel_mean_luma": "30",
                    "stage220_frame_lowlight_proxy": "true", "stage220_frame_night_candidate": "true",
                })
            output = root / "output"
            argv = [
                str(SCRIPT), "--manifest", str(manifest),
                "--expected-manifest-sha256", MODULE.sha256_file(manifest),
                "--output-root", str(output), "--per-sheet", "30", "--columns", "5",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)
            report = json.loads((output / "stage221-contact-sheet-index.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rows"], 1)
            self.assertEqual(len(report["sheets"]), 1)
            self.assertTrue((output / "stage221-contact-sheet-01.png").is_file())


if __name__ == "__main__":
    unittest.main()
