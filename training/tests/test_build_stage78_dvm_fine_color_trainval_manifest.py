import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_stage78_dvm_fine_color_trainval_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("stage78_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def source(split="train", source_color="Grey", color="silver_gray"):
    return {
        "image_path": "pool/a.jpg",
        "split": split,
        "body_type": "sedan",
        "body_type_supervised": "true",
        "color": color,
        "color_supervised": "true",
        "source_dataset": "DVM-CAR-2.0",
        "source_color": source_color,
        "source_license": "CC-BY-NC-4.0",
        "source_group": f"ad-{split}",
        "sha256": "a" * 64,
    }


class TransformTests(unittest.TestCase):
    def test_preserves_validation_and_maps_official_fine_color(self):
        row, error = MODULE.transform_row(source(split="validation"), "b" * 64)
        self.assertIsNone(error)
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["color"], "gray")
        self.assertEqual(row["body_type"], "unknown")
        self.assertEqual(row["body_type_supervised"], "false")

    def test_gray_silver_are_separated_only_by_official_metadata(self):
        gray, _ = MODULE.transform_row(source(source_color="Grey"), "b" * 64)
        silver, _ = MODULE.transform_row(source(source_color="Silver"), "b" * 64)
        self.assertEqual(gray["color"], "gray")
        self.assertEqual(silver["color"], "silver")

    def test_rejects_merged_label_conflict(self):
        row, error = MODULE.transform_row(source(color="gray"), "b" * 64)
        self.assertIsNone(row)
        self.assertIn("conflicts", error)

    def test_rejects_test_and_frozen_rows(self):
        row, error = MODULE.transform_row(source(split="test"), "b" * 64)
        self.assertIsNone(row)
        self.assertIn("test", error)
        frozen = source()
        frozen["image_path"] = "vcas_rtsp_demo_60s/frame.jpg"
        row, error = MODULE.transform_row(frozen, "b" * 64)
        self.assertIsNone(row)
        self.assertIn("frozen", error)


if __name__ == "__main__":
    unittest.main()
