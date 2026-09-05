import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage197", ROOT / "scripts" / "inspect_stage197_unknown_scene_scope.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["stage197"] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage197ScopeTest(unittest.TestCase):
    def test_path_template_masks_numeric_runs(self):
        self.assertEqual(
            MODULE.path_template("/data/cam2/frame_00123_car_9.jpg"),
            "/data/cam2/frame_#_car_#.jpg",
        )

    def test_truthy_fails_closed(self):
        self.assertTrue(MODULE.truthy("TRUE"))
        self.assertFalse(MODULE.truthy("unknown"))


if __name__ == "__main__":
    unittest.main()
