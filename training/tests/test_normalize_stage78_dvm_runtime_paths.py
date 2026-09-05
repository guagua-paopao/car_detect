from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize_stage78_dvm_runtime_paths.py"
SPEC = importlib.util.spec_from_file_location("normalize_stage78", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NormalizeStage78Tests(unittest.TestCase):
    def fixture(self, root: Path, *, frozen: bool = False, bad_sha: bool = False) -> Namespace:
        safety = root / "datasets"
        source = safety / "stage68"
        image = source / "crops" / "train" / "gray" / "x.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        labels = root / "labels.json"
        labels.write_text(json.dumps({
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["unknown"],
            "colors": ["gray", "unknown"],
        }), encoding="utf-8")
        manifest = root / "input.csv"
        fields = ["image_path", "split", "color", "color_supervised", "review_status", "crop_sha256", "source_frame_id"]
        row = {
            "image_path": "crops/train/gray/x.jpg",
            "split": "train",
            "color": "gray",
            "color_supervised": "true",
            "review_status": "approved",
            "crop_sha256": "0" * 64 if bad_sha else file_sha(image),
            "source_frame_id": "vcas_rtsp_demo_60s" if frozen else "source-1",
        }
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        report = root / "input-report.json"
        report.write_text(json.dumps({
            "status": "pass",
            "output": {"rows": 1, "sha256": file_sha(manifest)},
            "policy": {
                "test_rows_imported": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "deployment_performed": False,
            },
        }), encoding="utf-8")
        return Namespace(
            input_manifest=manifest,
            expected_input_manifest_sha256=file_sha(manifest),
            input_report=report,
            expected_input_report_sha256=file_sha(report),
            labels=labels,
            expected_labels_sha256=file_sha(labels),
            source_root=source,
            datasets_safety_root=safety,
            output_manifest=root / "out" / "runtime.csv",
            output_report=root / "out" / "runtime-report.json",
        )

    def test_normalizes_only_path_and_verifies_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            result = MODULE.normalize(args)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["integrity"]["non_path_projection_unchanged"])
            with args.output_manifest.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertTrue(Path(row["image_path"]).is_absolute())
            self.assertEqual(row["color"], "gray")

    def test_rejects_crop_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), bad_sha=True)
            with self.assertRaisesRegex(RuntimeError, "crop SHA256 mismatch"):
                MODULE.normalize(args)

    def test_rejects_frozen_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), frozen=True)
            with self.assertRaisesRegex(RuntimeError, "frozen marker"):
                MODULE.normalize(args)


if __name__ == "__main__":
    unittest.main()
