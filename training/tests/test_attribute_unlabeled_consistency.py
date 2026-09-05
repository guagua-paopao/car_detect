from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_attribute.py"
SPEC = importlib.util.spec_from_file_location("train_attribute", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path",
    "split",
    "body_type",
    "body_type_supervised",
    "color",
    "color_supervised",
    "night",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row() -> dict[str, str]:
    return {
        "image_path": "crops/example.jpg",
        "split": "train",
        "body_type": "unknown",
        "body_type_supervised": "false",
        "color": "unknown",
        "color_supervised": "false",
        "night": "true",
    }


class FailClosedUnlabeledManifestTests(unittest.TestCase):
    def test_accepts_only_explicit_unknown_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "unlabeled.csv"
            write_manifest(manifest, [valid_row()])
            rows = MODULE.load_fail_closed_unlabeled_rows(manifest)
            self.assertEqual(len(rows), 1)
            self.assertTrue(MODULE.truthy(rows[0]["night"]))

    def test_rejects_evaluation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "unlabeled.csv"
            row = valid_row()
            row["split"] = "validation"
            write_manifest(manifest, [row])
            with self.assertRaisesRegex(ValueError, "train rows only"):
                MODULE.load_fail_closed_unlabeled_rows(manifest)

    def test_rejects_supervised_or_known_targets(self) -> None:
        for field, value in (("body_type_supervised", "true"), ("color", "red")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory) / "unlabeled.csv"
                row = valid_row()
                row[field] = value
                write_manifest(manifest, [row])
                with self.assertRaises(ValueError):
                    MODULE.load_fail_closed_unlabeled_rows(manifest)

    def test_resolves_sibling_dataset_relative_to_manifest_with_common_safety_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            manifest = root / "attribute-domain-v2" / "unlabeled.csv"
            manifest.parent.mkdir(parents=True)
            expected = root / "sibling-dataset" / "crops" / "example.jpg"
            resolved = MODULE.resolve_unlabeled_image_path(
                manifest,
                root,
                "../sibling-dataset/crops/example.jpg",
            )
            self.assertEqual(resolved, expected.resolve())

    def test_rejects_unlabeled_image_outside_common_safety_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            manifest = root / "attribute-domain-v2" / "unlabeled.csv"
            manifest.parent.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "escapes safety root"):
                MODULE.resolve_unlabeled_image_path(
                    manifest,
                    root,
                    "../../outside.jpg",
                )


if __name__ == "__main__":
    unittest.main()
