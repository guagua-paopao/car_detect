from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage107_test_views.py"
SPEC = importlib.util.spec_from_file_location("stage107_views", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage107ViewsTest(unittest.TestCase):
    def write_manifest(self, root: Path, split: str = "test") -> tuple[Path, Path]:
        image = root / "source" / "image.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"fixture")
        manifest = root / "source" / "manifest.csv"
        fields = [
            "image_path",
            "split",
            "review_status",
            "body_type",
            "body_type_supervised",
            "color",
            "color_supervised",
            "track_key",
            "source_video",
            "crop_sha256",
        ]
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": image.name,
                    "split": split,
                    "review_status": "approved",
                    "body_type": "sedan",
                    "body_type_supervised": "true",
                    "color": "blue",
                    "color_supervised": "true",
                    "track_key": "track-1",
                    "source_video": "video-1",
                    "crop_sha256": "a" * 64,
                }
            )
        return manifest, image

    def test_build_view_preserves_test_membership_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, image = self.write_manifest(root)
            output = root / "out" / "hard.test.csv"
            report, digests = MODULE.build_view("hard", manifest, output, root)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["split"], "test")
            self.assertEqual(rows[0]["body_type"], "sedan")
            self.assertEqual(rows[0]["color"], "blue")
            self.assertEqual((output.parent / rows[0]["image_path"]).resolve(), image.resolve())
            self.assertFalse(report["labels_modified"])
            self.assertEqual(digests, {"a" * 64})

    def test_source_without_test_rows_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.write_manifest(root, split="validation")
            with self.assertRaisesRegex(RuntimeError, "no test rows"):
                MODULE.build_view("hard", manifest, root / "out" / "hard.csv", root)

    def test_frozen_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.write_manifest(root)
            text = manifest.read_text(encoding="utf-8").replace("video-1", "36-48")
            manifest.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen marker"):
                MODULE.build_view("hard", manifest, root / "out" / "hard.csv", root)


if __name__ == "__main__":
    unittest.main()
