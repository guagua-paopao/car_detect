import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage195", ROOT / "scripts" / "audit_stage195_abtd_train_images.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["stage195"] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage195Test(unittest.TestCase):
    def test_source_group_buckets_consecutive_frames(self):
        self.assertEqual(MODULE.source_group("FOGGY_NIGHT_clip_001"), MODULE.source_group("FOGGY_NIGHT_clip_004"))
        self.assertNotEqual(MODULE.source_group("FOGGY_NIGHT_clip_004"), MODULE.source_group("FOGGY_NIGHT_clip_005"))

    def test_exact_mapping_excludes_coarse_and_regional_classes(self):
        self.assertEqual(MODULE.EXACT_BODY, {1: "bus", 2: "truck"})
        for class_id in (0, 3, 4, 5, 6, 7, 8, 9):
            self.assertNotIn(class_id, MODULE.EXACT_BODY)


if __name__ == "__main__":
    unittest.main()
