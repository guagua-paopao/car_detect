import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_stage219_color_inventory_to_active.py"
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name("compare_stage219_color_inventory_to_active.py")
SPEC = importlib.util.spec_from_file_location("stage219", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage219DeltaTests(unittest.TestCase):
    def test_sha_key_preferred(self):
        sha = "b" * 64
        self.assertEqual(MODULE.row_key({"sha256": sha, "image_path": "x.jpg"}, Path("m.csv")), ("sha256", sha))

    def test_relative_path_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "m.csv"
            expected = ("path", os.path.normcase(os.path.normpath(str(Path(temporary) / "x.jpg"))))
            self.assertEqual(MODULE.row_key({"image_path": "x.jpg"}, manifest), expected)

    def test_frozen_marker(self):
        self.assertTrue(MODULE.has_frozen_marker({"video_id": "vcas_rtsp_demo_60s"}))
        self.assertFalse(MODULE.has_frozen_marker({"video_id": "train-camera-1"}))


if __name__ == "__main__":
    unittest.main()
