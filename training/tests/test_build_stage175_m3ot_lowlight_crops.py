from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage175_m3ot_lowlight_crops.py"
SPEC = importlib.util.spec_from_file_location("stage175", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage175BuilderTests(unittest.TestCase):
    def test_uniform_sampling_is_bounded_and_keeps_ends(self) -> None:
        records = [{"mot_frame_id": value, "annotation_id": value} for value in range(10)]
        sampled = MODULE.uniformly_sample(records, 4)
        self.assertEqual([row["mot_frame_id"] for row in sampled], [0, 3, 6, 9])

    def test_square_crop_rejects_invalid_bbox(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        with self.assertRaises(ValueError):
            MODULE.square_crop(image, [1, 1, -2, 3], 1.6, 32)

    def test_dhash_is_stable(self) -> None:
        image = Image.new("RGB", (32, 32), "black")
        self.assertEqual(MODULE.dhash64(image), MODULE.dhash64(image.copy()))


if __name__ == "__main__":
    unittest.main()
