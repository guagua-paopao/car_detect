from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage92_lvad_archive.py"
SPEC = importlib.util.spec_from_file_location("audit_stage92_lvad_archive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage92LVADArchiveAuditTests(unittest.TestCase):
    def test_safe_member_rejects_traversal_absolute_and_backslash(self) -> None:
        self.assertTrue(MODULE.safe_member("root/train/images/1.jpg"))
        self.assertFalse(MODULE.safe_member("../escape.jpg"))
        self.assertFalse(MODULE.safe_member("/absolute.jpg"))
        self.assertFalse(MODULE.safe_member("root\\escape.jpg"))

    def test_numeric_classes_are_conservatively_mapped(self) -> None:
        self.assertEqual(MODULE.CLASS_CONTRACT[0], "motorcycle_excluded")
        self.assertIn("unknown_only", MODULE.CLASS_CONTRACT[1])
        self.assertIn("coarse_only", MODULE.CLASS_CONTRACT[2])

    def test_parse_yolo_accepts_three_classes_and_counts_small(self) -> None:
        counts, small = MODULE.parse_yolo("0 0.5 0.5 0.02 0.02\n2 0.5 0.5 0.5 0.5\n", 1000, 500)
        self.assertEqual(counts, {0: 1, 2: 1})
        self.assertEqual(small, 1)

    def test_parse_yolo_rejects_unknown_and_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("3 0.5 0.5 0.2 0.2\n", 100, 100)
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("1 0.99 0.5 0.2 0.2\n", 100, 100)

    def test_out_of_contract_class_is_counted_for_frame_exclusion(self) -> None:
        self.assertEqual(MODULE.out_of_contract_classes("1 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n"), {3: 1})
        self.assertLessEqual(1, MODULE.MAX_INVALID_CLASS_FRAMES)

    def test_source_frame_key_strips_only_roboflow_hash(self) -> None:
        self.assertEqual(MODULE.source_frame_key("frame1_000123_jpg.rf.abcdef"), "frame1_000123")
        self.assertEqual(MODULE.source_frame_key("frame1_000123"), "frame1_000123")

    def test_split_contract_keeps_heldout_payloads_separate(self) -> None:
        self.assertEqual(MODULE.EXPECTED_IMAGES, {"train": 10934, "valid": 1364, "test": 1365})
        self.assertEqual(MODULE.EXPECTED_LABELS["train"], 10919)


if __name__ == "__main__":
    unittest.main()
