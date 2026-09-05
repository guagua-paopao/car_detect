from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "rewrite_stage175_cloud_paths.py"
SPEC = importlib.util.spec_from_file_location("stage175_paths", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage175PathTests(unittest.TestCase):
    def test_windows_absolute_crop_becomes_cloud_relative(self) -> None:
        value = r"F:\codex\project\training\datasets\stage175\crops\image.jpg"
        self.assertEqual(MODULE.rewrite_path(value, "crops"), "crops/image.jpg")

    def test_unexpected_parent_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.rewrite_path(r"F:\unsafe\image.jpg", "crops")


if __name__ == "__main__":
    unittest.main()
