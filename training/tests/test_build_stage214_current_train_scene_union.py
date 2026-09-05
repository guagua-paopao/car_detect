from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage214_current_train_scene_union.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage214UnionTests(unittest.TestCase):
    def test_union_is_train_only_and_reports_cross_role_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image1 = root / "a.jpg"
            image2 = root / "b.jpg"
            image1.write_bytes(b"a")
            image2.write_bytes(b"b")
            fields = ["image_path", "split", "review_status", "body_type", "color"]
            supervised = root / "supervised.csv"
            unlabeled = root / "unlabeled.csv"
            with supervised.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"image_path": image1, "split": "train", "review_status": "approved", "body_type": "sedan", "color": "unknown"})
                writer.writerow({"image_path": image2, "split": "validation", "review_status": "approved", "body_type": "suv", "color": "unknown"})
            with unlabeled.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"image_path": image1, "split": "train", "review_status": "pending", "body_type": "unknown", "color": "unknown"})
                writer.writerow({"image_path": image2, "split": "train", "review_status": "pending", "body_type": "unknown", "color": "unknown"})
            output = root / "out"
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--supervised-manifest", str(supervised),
                "--expected-supervised-sha256", sha256(supervised),
                "--unlabeled-manifest", str(unlabeled),
                "--expected-unlabeled-sha256", sha256(unlabeled),
                "--allowed-root", str(root),
                "--output-root", str(output),
            ], check=True, capture_output=True, text=True)
            report = json.loads((output / "stage214-current-train-scene-union-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["union_rows"], 3)
            self.assertEqual(report["counts"]["cross_role_path_overlap"], 1)
            self.assertEqual(report["counts"]["combined_unique_paths"], 2)
            self.assertEqual(report["counts"]["validation_or_test_rows_written"], 0)

    def test_frozen_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            fields = ["image_path", "split", "review_status", "body_type", "color", "video_id"]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"image_path": root / "a.jpg", "split": "train", "review_status": "approved", "body_type": "sedan", "color": "unknown", "video_id": "vcas_rtsp_demo_60s"})
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--supervised-manifest", str(manifest),
                "--expected-supervised-sha256", sha256(manifest),
                "--unlabeled-manifest", str(manifest),
                "--expected-unlabeled-sha256", sha256(manifest),
                "--allowed-root", str(root),
                "--output-root", str(root / "out"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen-video marker", result.stderr)


if __name__ == "__main__":
    unittest.main()
