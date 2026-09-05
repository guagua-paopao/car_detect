from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "build_stage102_v2_partial_color_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_stage102", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path", "split", "review_status", "body_type", "color",
    "body_type_supervised", "color_supervised", "formal_train_eligible",
    "license_train_eligible", "source_dataset", "source_group", "track_group",
    "source_license", "annotation_source",
    "sha256", "dhash64", "stage71_origin", "stage74_joint_source",
    "pseudo_label", "night", "low_light", "small_target", "vehicle_size",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides: str) -> dict[str, str]:
    value = {
        "image_path": "image.jpg", "split": "train", "review_status": "approved",
        "body_type": "unknown", "color": "black", "body_type_supervised": "false",
        "color_supervised": "true", "formal_train_eligible": "true",
        "license_train_eligible": "true", "source_dataset": "UA-DETRAC",
        "source_license": "CC BY-NC research-only", "annotation_source": "teacher_consensus",
        "source_group": "source-a", "track_group": "track-a", "sha256": "a" * 64,
        "dhash64": "0000000000000000", "stage71_origin": "cctv",
        "stage74_joint_source": "ua_detrac", "pseudo_label": "true",
        "night": "true", "low_light": "true", "small_target": "true",
        "vehicle_size": "small",
    }
    value.update(overrides)
    return value


class Stage102PartialColorManifestTest(unittest.TestCase):
    def test_merged_silver_gray_becomes_partial_group_not_exact_class(self) -> None:
        converted = MODULE.convert_cctv_row(row(color="silver_gray"))
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["color"], "unknown")
        self.assertEqual(converted["color_supervised"], "false")
        self.assertEqual(converted["coarse_color_group"], "silver_gray")

    def test_validation_or_ineligible_rows_are_never_selected(self) -> None:
        self.assertIsNone(MODULE.convert_cctv_row(row(split="validation")))
        self.assertIsNone(MODULE.convert_cctv_row(row(formal_train_eligible="false")))
        self.assertIsNone(MODULE.convert_cctv_row(row(formal_train_eligible="")))
        self.assertIsNone(MODULE.convert_cctv_row(row(license_train_eligible="")))
        self.assertIsNone(MODULE.convert_cctv_row(row(source_license="")))
        self.assertIsNone(MODULE.convert_cctv_row(row(review_status="")))
        self.assertIsNone(MODULE.convert_cctv_row(row(color_supervised="false")))

    def test_build_preserves_base_validation_and_filters_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.csv"
            cctv = root / "cctv.csv"
            base_rows = [
                row(
                    image_path="base-train.jpg", split="train", source_dataset="DVM-CAR",
                    source_group="dvm-train", track_group="", sha256="1" * 64,
                    dhash64="1111111111111111", stage71_origin="", stage74_joint_source="",
                    color="gray",
                ),
                row(
                    image_path="base-val.jpg", split="validation", source_dataset="DVM-CAR",
                    source_group="dvm-validation", track_group="", sha256="2" * 64,
                    dhash64="2222222222222222", stage71_origin="", stage74_joint_source="",
                    color="silver",
                ),
            ]
            cctv_rows = [
                row(image_path="black.jpg", sha256="3" * 64, dhash64="3333333333333333", color="black"),
                row(
                    image_path="silver-gray.jpg", sha256="4" * 64,
                    dhash64="4444444444444444", color="silver_gray",
                    source_group="source-b", track_group="track-b",
                ),
                row(image_path="duplicate.jpg", sha256="2" * 64, color="black"),
                row(image_path="test.jpg", split="test", sha256="5" * 64, color="red"),
            ]
            write_manifest(base, base_rows)
            write_manifest(cctv, cctv_rows)
            fields, output, report = MODULE.build(
                base, cctv, root, near_distance=0, audit_images=False,
                minimum_cctv_rows=2, minimum_groups=2,
            )
            self.assertIn("coarse_color_group", fields)
            self.assertEqual(len(output), 4)
            self.assertEqual(sum(item["split"] == "validation" for item in output), 1)
            added = [item for item in output if item.get("stage102_origin")]
            self.assertEqual({item.get("color") for item in added}, {"black", "unknown"})
            self.assertEqual(report["rows"]["added_cctv_train"], 2)
            self.assertEqual(report["rejections"]["exact_duplicate"], 1)
            self.assertEqual(report["added_license_counts"], {"CC BY-NC research-only": 2})
            self.assertFalse(report["policy"]["frozen_video_used"])


if __name__ == "__main__":
    unittest.main()
