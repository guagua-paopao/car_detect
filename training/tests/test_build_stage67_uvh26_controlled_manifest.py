from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage67_uvh26_controlled_manifest.py"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage67UVH26ControlledManifestTest(unittest.TestCase):
    def test_builds_group_capped_type_only_supplement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fields = [
                "image_path", "body_type", "color", "split", "track_group", "source_frame_id",
                "review_status", "body_type_supervised", "color_supervised", "source_dataset",
                "source_license", "license_train_eligible", "formal_train_eligible", "crop_sha256",
                "dhash64", "vehicle_size", "small_target",
            ]
            base_image = root / "base.png"; Image.new("RGB", (16, 16), (10, 20, 30)).save(base_image)
            heldout_image = root / "heldout.png"; Image.new("RGB", (16, 16), (200, 20, 30)).save(heldout_image)
            base = root / "base.csv"
            base_rows = [
                {"image_path": str(base_image), "body_type": "sedan", "color": "unknown", "split": "train", "track_group": "base", "source_frame_id": "base", "review_status": "approved", "body_type_supervised": "true", "color_supervised": "false", "source_dataset": "base", "source_license": "CC-BY-4.0", "license_train_eligible": "true", "formal_train_eligible": "true", "crop_sha256": file_sha(base_image), "dhash64": "ffffffffffffffff", "vehicle_size": "large", "small_target": "false"},
                {"image_path": str(heldout_image), "body_type": "sedan", "color": "unknown", "split": "validation", "track_group": "heldout", "source_frame_id": "heldout", "review_status": "approved", "body_type_supervised": "true", "color_supervised": "false", "source_dataset": "base", "source_license": "CC-BY-4.0", "license_train_eligible": "true", "formal_train_eligible": "true", "crop_sha256": file_sha(heldout_image), "dhash64": "0000000000000000", "vehicle_size": "large", "small_target": "false"},
            ]
            with base.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(base_rows)
            leakage = root / "base-leakage.json"
            leakage.write_text(json.dumps({"status": "pass", "manifest_sha256": file_sha(base), "dhash": {"cross_split_near_pairs": 0}}), encoding="utf-8")
            uvh = root / "uvh.csv"; uvh_rows = []
            for index, body_type in enumerate(("sedan", "suv", "bus", "van")):
                image = root / f"uvh-{index}.png"; Image.new("RGB", (16, 16), (30 + index * 20, 100, 50)).save(image)
                uvh_rows.append({"image_path": str(image), "body_type": body_type, "color": "unknown", "split": "train", "track_group": f"g-{index // 2}", "source_frame_id": f"f-{index // 2}", "review_status": "approved", "body_type_supervised": "true", "color_supervised": "false", "source_dataset": "UVH-26", "source_license": "CC-BY-4.0", "license_train_eligible": "", "formal_train_eligible": "", "crop_sha256": file_sha(image), "dhash64": "", "vehicle_size": "small", "small_target": "true"})
            with uvh.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(uvh_rows)
            uvh_report = root / "uvh-report.json"
            uvh_report.write_text(json.dumps({"source_dataset": "iisc-aim/UVH-26", "source_license": "CC-BY-4.0", "output_manifest_sha256": file_sha(uvh)}), encoding="utf-8")
            license_evidence = root / "license.json"
            license_evidence.write_text(json.dumps({"status": "pass_primary_source_verified", "declared_dataset_license": "CC-BY-4.0", "primary_source_findings": {"training_use_allowed_with_attribution": True}, "decision": {"production_training_license_eligible": True}}), encoding="utf-8")
            output = root / "output.csv"; report = root / "report.json"
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--base-manifest", str(base),
                "--base-leakage-report", str(leakage), "--uvh-manifest", str(uvh),
                "--uvh-report", str(uvh_report), "--license-evidence", str(license_evidence),
                "--output-manifest", str(output), "--output-report", str(report),
                "--maximum-rows", "4", "--maximum-per-group", "2", "--workers", "1",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(evidence["selected_uvh_rows"], 4)
            self.assertEqual(evidence["selected_unique_groups"], 2)
            self.assertEqual(evidence["selected_maximum_rows_per_group"], 2)
            with output.open("r", encoding="utf-8", newline="") as handle:
                selected = [row for row in csv.DictReader(handle) if row.get("stage67_uvh26_licensed") == "true"]
            self.assertTrue(all(row["license_train_eligible"] == "true" for row in selected))
            self.assertTrue(all(row["color"] == "unknown" and row["color_supervised"] == "false" for row in selected))
            self.assertTrue(all(len(row["dhash64"]) == 16 for row in selected))


if __name__ == "__main__":
    unittest.main()
