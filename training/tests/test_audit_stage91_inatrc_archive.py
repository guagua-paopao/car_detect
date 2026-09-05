from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage91_inatrc_archive.py"
SPEC = importlib.util.spec_from_file_location("audit_stage91_inatrc_archive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage91InaTRCArchiveAuditTests(unittest.TestCase):
    def test_safe_member_rejects_traversal_absolute_and_backslash(self) -> None:
        self.assertTrue(MODULE.safe_member("root/train/images/1.jpg"))
        self.assertFalse(MODULE.safe_member("../escape.jpg"))
        self.assertFalse(MODULE.safe_member("/absolute.jpg"))
        self.assertFalse(MODULE.safe_member("root\\escape.jpg"))

    def test_parse_yolo_accepts_official_zero_based_classes(self) -> None:
        counts, small = MODULE.parse_yolo("0 0.5 0.5 0.2 0.2\n4 0.5 0.5 0.5 0.5\n", 1000, 500)
        self.assertEqual(counts, {0: 1, 4: 1})
        self.assertEqual(small, 0)

    def test_parse_yolo_rejects_unknown_class_and_out_of_bounds_box(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("5 0.5 0.5 0.2 0.2\n", 100, 100)
        with self.assertRaises(ValueError):
            MODULE.parse_yolo("0 0.99 0.5 0.2 0.2\n", 100, 100)

    def test_expected_split_contract_keeps_test_separate(self) -> None:
        self.assertEqual(MODULE.EXPECTED_SPLIT_IMAGES, {"train": 2600, "val": 325, "test": 325})


if __name__ == "__main__":
    unittest.main()
