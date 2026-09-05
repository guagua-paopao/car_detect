from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage87_supervised_scene_quotas.py"
SPEC = importlib.util.spec_from_file_location("stage87_quota_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
audit_manifest = MODULE.audit_manifest


FIELDS = [
    "image_path",
    "body_type",
    "body_type_supervised",
    "split",
    "review_status",
    "crop_sha256",
    "night",
    "low_light",
    "lighting",
    "small_target",
    "vehicle_size",
    "occluded",
    "truncated",
    "occlusion_level",
    "hard_mining_tags",
    "source_dataset",
    "source_license",
    "track_group",
    "camera_id",
    "video_id",
    "sample_weight",
]


def base_row(index: int) -> dict[str, str]:
    return {
        "image_path": f"crops/{index}.jpg",
        "body_type": "sedan",
        "body_type_supervised": "true",
        "split": "train",
        "review_status": "approved",
        "crop_sha256": f"{index:064x}",
        "night": "false",
        "low_light": "false",
        "lighting": "daylight",
        "small_target": "false",
        "vehicle_size": "medium",
        "occluded": "false",
        "truncated": "false",
        "occlusion_level": "visible",
        "hard_mining_tags": "",
        "source_dataset": "fixture",
        "source_license": "CC-BY-4.0",
        "track_group": f"track-{index}",
        "camera_id": "camera-1",
        "video_id": "video-1",
        "sample_weight": "1.0",
    }


class Stage87QuotaAuditTest(unittest.TestCase):
    def write_fixture(self, root: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        labels = root / "labels.json"
        labels.write_text(json.dumps({"body_types": ["sedan", "suv", "unknown"]}), encoding="utf-8")
        return manifest, labels

    def test_counts_unique_rows_and_passes_exact_quota_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [base_row(index) for index in range(20)]
            for row in rows[:6]:
                row["night"] = "true"
            for row in rows[:4]:
                row["vehicle_size"] = "small"
            for row in rows[:3]:
                row["truncated"] = "true"
            duplicate = dict(rows[0])
            duplicate["image_path"] = "crops/duplicate.jpg"
            duplicate["sample_weight"] = "20.0"
            rows.append(duplicate)
            manifest, labels = self.write_fixture(root, rows)
            report = audit_manifest(manifest, labels)
            summary = report["summary"]
            self.assertEqual(summary["selected_rows_before_exact_dedup"], 21)
            self.assertEqual(summary["unique_loader_usable_supervised_train_rows"], 20)
            self.assertEqual(summary["exact_duplicate_rows_excluded_from_quota"], 1)
            self.assertTrue(summary["all_scene_quotas_pass"])

    def test_weight_does_not_fill_missing_unique_night_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [base_row(index) for index in range(10)]
            rows[0]["night"] = "true"
            rows[0]["sample_weight"] = "20.0"
            for row in rows[:2]:
                row["small_target"] = "true"
            for row in rows[:2]:
                row["occluded"] = "true"
            manifest, labels = self.write_fixture(root, rows)
            report = audit_manifest(manifest, labels)
            night = report["summary"]["quota"]["night_or_low_light"]
            self.assertEqual(night["unique_rows"], 1)
            self.assertEqual(night["additional_unique_rows_required"], 2)
            self.assertEqual(night["minimum_positive_only_additions_to_reach_final_fraction"], 3)
            self.assertFalse(night["pass"])

    def test_unknown_scene_metadata_is_not_counted_as_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [base_row(index) for index in range(10)]
            for row in rows:
                row["night"] = "unknown"
                row["small_target"] = "unknown"
                row["occluded"] = "unknown"
            manifest, labels = self.write_fixture(root, rows)
            report = audit_manifest(manifest, labels)
            quota = report["summary"]["quota"]
            self.assertEqual(quota["night_or_low_light"]["unique_rows"], 0)
            self.assertEqual(quota["small_target"]["unique_rows"], 0)
            self.assertEqual(quota["occluded_overlapped_or_truncated"]["unique_rows"], 0)

    def test_excludes_unapproved_unsupervised_validation_and_unknown_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [base_row(index) for index in range(20)]
            for row in rows[:6]:
                row["low_light"] = "true"
            for row in rows[:4]:
                row["small_target"] = "true"
            for row in rows[:3]:
                row["occlusion_level"] = "partial"
            rejected = base_row(100)
            rejected["review_status"] = "rejected_to_unknown"
            unsupervised = base_row(101)
            unsupervised["body_type_supervised"] = "false"
            validation = base_row(102)
            validation["split"] = "validation"
            unknown = base_row(103)
            unknown["body_type"] = "unknown"
            rows.extend([rejected, unsupervised, validation, unknown])
            manifest, labels = self.write_fixture(root, rows)
            report = audit_manifest(manifest, labels)
            self.assertEqual(report["summary"]["unique_loader_usable_supervised_train_rows"], 20)

    def test_fails_closed_on_frozen_video_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = base_row(1)
            row["image_path"] = "vcas_rtsp_demo_60s/frame.jpg"
            manifest, labels = self.write_fixture(root, [row])
            with self.assertRaisesRegex(ValueError, "frozen-video"):
                audit_manifest(manifest, labels)


if __name__ == "__main__":
    unittest.main()
