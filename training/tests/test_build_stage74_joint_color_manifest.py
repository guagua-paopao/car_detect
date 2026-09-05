from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.scripts.build_stage74_joint_color_manifest import (
    dhash64,
    has_near,
    has_near_any_orientation,
    projection_digest,
)


class Stage74JointManifestTest(unittest.TestCase):
    def test_near_duplicate_radius_is_exact(self) -> None:
        candidates = [0b0000, 0b1111]
        self.assertTrue(has_near(0b0011, candidates, 2))
        self.assertFalse(has_near(0b0111, [0b0000], 2))
        value = int("0000000000000000", 16)
        complement = int("ffffffffffffffff", 16)
        self.assertTrue(has_near_any_orientation(value, [complement], 0))

    def test_projection_digest_ignores_new_fields_but_not_validation_changes(self) -> None:
        fields = ["image_path", "color"]
        before = [{"image_path": "a.jpg", "color": "red"}]
        after = [{"image_path": "a.jpg", "color": "red", "new": "x"}]
        changed = [{"image_path": "a.jpg", "color": "blue", "new": "x"}]
        self.assertEqual(projection_digest(before, fields), projection_digest(after, fields))
        self.assertNotEqual(projection_digest(before, fields), projection_digest(changed, fields))

    def test_dhash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient.png"
            image = Image.new("L", (9, 8))
            image.putdata([x for _y in range(8) for x in range(9)])
            image.save(path)
            self.assertEqual(dhash64(path), dhash64(path))
            self.assertEqual(len(dhash64(path)), 16)

    def test_script_pins_new_rows_to_cctv_weight_contract(self) -> None:
        text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_stage74_joint_color_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn('row["stage71_origin"] = "cctv"', text)
        self.assertIn('row["sample_weight"] = "10.000000"', text)
        self.assertIn('row["sample_weight"] = "0.300000"', text)


if __name__ == "__main__":
    unittest.main()
