import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "resolve_stage220_novel_adverse_color_candidates.py"
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name("resolve_stage220_novel_adverse_color_candidates.py")
SPEC = importlib.util.spec_from_file_location("stage220", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage220ResolutionTests(unittest.TestCase):
    def test_authoritative_rejects_teacher(self):
        base = {"color": "red", "color_supervised": "true"}
        self.assertTrue(MODULE.authoritative_color({**base, "review_method": "official_registration_color"}))
        self.assertFalse(MODULE.authoritative_color({**base, "review_method": "teacher_consensus"}))

    def test_skip_test_manifest(self):
        self.assertTrue(MODULE.skip_manifest(Path("/x/independent-test/manifest.csv")))
        self.assertFalse(MODULE.skip_manifest(Path("/x/train/manifest.csv")))

    def test_dhash_is_deterministic(self):
        gray = np.arange(72, dtype=np.uint8).reshape(8, 9)
        self.assertEqual(MODULE.dhash64(gray), MODULE.dhash64(gray.copy()))

    def test_resolved_image_stays_within_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = root / "manifests" / "m.csv"
            image = root / "images" / "a.jpg"
            manifest.parent.mkdir()
            image.parent.mkdir()
            image.write_bytes(b"x")
            self.assertEqual(MODULE.resolved_image_path("../images/a.jpg", manifest, root), image)


if __name__ == "__main__":
    unittest.main()
