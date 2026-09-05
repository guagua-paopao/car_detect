from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_stage91_inatrc_train_crops.py"
SPEC = importlib.util.spec_from_file_location("build_stage91_inatrc_train_crops", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage91InaTRCTrainCropTests(unittest.TestCase):
    def test_mixed_nontruck_is_never_fine_supervised(self) -> None:
        self.assertEqual(MODULE.CLASS_MAPPING[0], ("unknown", False, "mixed_nontruck_unknown"))
        self.assertEqual(MODULE.CLASS_MAPPING[1][0], "truck")
        self.assertEqual(MODULE.CLASS_MAPPING[4][0], "heavy_truck")

    def test_parse_yolo_accepts_zero_based_contract(self) -> None:
        boxes = MODULE.parse_yolo("0 0.5 0.5 0.2 0.2\n4 0.4 0.4 0.1 0.1\n")
        self.assertEqual(len(boxes), 2)
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("5 0.5 0.5 0.2 0.2\n")

    def test_crop_box_clamps_padding_and_rejects_tiny_crop(self) -> None:
        image = Image.new("RGB", (100, 100), "gray")
        crop = MODULE.crop_box(image, (0.05, 0.05, 0.1, 0.1), 0.1)
        self.assertGreaterEqual(crop.width, 8)
        with self.assertRaises(ValueError):
            MODULE.crop_box(image, (0.5, 0.5, 0.01, 0.01), 0.0)

    def test_quality_gate_demotes_visually_unusable_small_crop(self) -> None:
        poor = Image.new("RGB", (20, 20), (30, 30, 30))
        self.assertFalse(MODULE.inspect_crop(poor)["quality_supervised"])


if __name__ == "__main__":
    unittest.main()
