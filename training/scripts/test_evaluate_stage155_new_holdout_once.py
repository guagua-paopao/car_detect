import csv
from pathlib import Path
import tempfile
import unittest

import evaluate_stage155_new_holdout_once as evaluator


class Stage155EvaluatorTests(unittest.TestCase):
    def write_manifest(self, path: Path, row: dict[str, str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_test_only_rows_are_resolved_inside_safety_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, {"split": "test", "review_status": "approved", "image_path": str(image)})
            rows = evaluator.resolve_test_rows(manifest, root)
            self.assertEqual(rows[0]["_resolved_image_path"], str(image.resolve()))

    def test_non_test_split_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, {"split": "validation", "review_status": "approved", "image_path": str(image)})
            with self.assertRaisesRegex(RuntimeError, "test-only"):
                evaluator.resolve_test_rows(manifest, root)

    def test_path_outside_safety_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            image = Path(outside) / "image.jpg"
            image.write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, {"split": "test", "review_status": "approved", "image_path": str(image)})
            with self.assertRaisesRegex(RuntimeError, "escapes datasets safety root"):
                evaluator.resolve_test_rows(manifest, root)


if __name__ == "__main__":
    unittest.main()
