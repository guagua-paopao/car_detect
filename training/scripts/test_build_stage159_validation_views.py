import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_stage159_validation_views import seal_view


class Stage159ViewTests(unittest.TestCase):
    def test_seal_view_keeps_only_supervised_validation_and_absolutizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "vehicle.jpg"
            Image.new("RGB", (8, 8), (100, 110, 120)).save(image)
            rows = [
                {"image_path": str(image), "split": "train", "review_status": "approved", "color_supervised": "true", "color": "blue", "track_group": "train"},
                {"image_path": str(image), "split": "validation", "review_status": "approved", "color_supervised": "true", "color": "blue", "track_group": "val"},
                {"image_path": str(image), "split": "validation", "review_status": "approved", "color_supervised": "false", "color": "unknown", "track_group": "unsup"},
            ]
            selected, summary = seal_view(rows, "color", root, (root,))
            self.assertEqual(len(selected), 1)
            self.assertTrue(Path(selected[0]["image_path"]).is_absolute())
            self.assertEqual(summary["label_counts"], {"blue": 1})

    def test_rejects_frozen_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "vehicle.jpg"
            Image.new("RGB", (8, 8)).save(image)
            rows = [{"image_path": str(image), "split": "validation", "review_status": "approved", "color_supervised": "true", "color": "blue", "track_group": "frozen_video"}]
            with self.assertRaisesRegex(RuntimeError, "forbidden marker"):
                seal_view(rows, "color", root, (root,))


if __name__ == "__main__":
    unittest.main()
