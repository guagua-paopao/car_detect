from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage92_lvad_night_pool.py"
SPEC = importlib.util.spec_from_file_location("build_stage92_lvad_night_pool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage92LVADNightPoolTests(unittest.TestCase):
    def test_class_mapping_never_creates_fine_truth(self) -> None:
        self.assertEqual(MODULE.CLASS_MAPPING[0][0], "exclude")
        self.assertEqual(MODULE.CLASS_MAPPING[1][0], "car")
        self.assertEqual(MODULE.CLASS_MAPPING[2][0], "truck")

    def test_frame_identity_preserves_sequence_and_numeric_order(self) -> None:
        self.assertEqual(MODULE.frame_identity("frame1_000123_jpg.rf.abcdef"), ("frame1", 123))
        self.assertEqual(MODULE.frame_identity("frame_009426_jpg.rf.abcdef"), ("frame", 9426))
        self.assertEqual(MODULE.frame_identity("frame_006317_jpg.rf.abcdef(1)"), ("frame", 6317))

    def test_temporal_gap_rejects_adjacent_frames(self) -> None:
        state: dict[str, int] = {}
        self.assertTrue(MODULE.temporal_keep(state, "frame1", 100, 15))
        self.assertFalse(MODULE.temporal_keep(state, "frame1", 110, 15))
        self.assertTrue(MODULE.temporal_keep(state, "frame1", 115, 15))

    def test_parse_yolo_rejects_out_of_contract_class(self) -> None:
        self.assertEqual(len(MODULE.parse_yolo("1 0.5 0.5 0.2 0.2\n")), 1)
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("3 0.5 0.5 0.2 0.2\n")

    def test_crop_quality_rejects_tiny_uniform_content(self) -> None:
        poor = Image.new("RGB", (12, 12), (20, 20, 20))
        self.assertFalse(MODULE.inspect_crop(poor)["usable"])

    def test_bk_tree_finds_near_hash(self) -> None:
        tree = MODULE.HammingBKTree()
        row = {"id": "a"}
        tree.add(0, row)
        self.assertEqual(tree.find(3, 2), row)
        self.assertIsNone(tree.find(7, 2))


if __name__ == "__main__":
    unittest.main()
