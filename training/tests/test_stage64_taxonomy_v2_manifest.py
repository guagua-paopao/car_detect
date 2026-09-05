from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "build_stage64_taxonomy_v2_manifest.py"
LABELS = TRAINING.parent / "config" / "vehicle_labels.v2.json"


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_row(**updates: str) -> dict[str, str]:
    row = {
        "image_path": "crop.jpg",
        "split": "train",
        "source_dataset": "VCoR-LARGE",
        "source_frame_id": "vcor:train/grey/example.jpg",
        "source_color": "",
        "color": "silver_gray",
        "color_supervised": "true",
        "body_type": "unknown",
        "body_type_supervised": "false",
        "official_vehicle_class": "",
    }
    row.update(updates)
    return row


class Stage64TaxonomyV2ManifestTest(unittest.TestCase):
    def run_builder(
        self,
        root: Path,
        rows: list[dict[str, str]],
        vcor_rows: list[dict[str, str]] | None = None,
    ) -> subprocess.CompletedProcess:
        source = root / "input.csv"
        write_manifest(source, rows)
        command = [
                sys.executable,
                str(SCRIPT),
                "--input-manifest",
                str(source),
                "--output-manifest",
                str(root / "output.csv"),
                "--output-report",
                str(root / "report.json"),
                "--labels",
                str(LABELS),
            ]
        if vcor_rows is not None:
            vcor_root = root / "vcor"
            vcor_root.mkdir()
            for row in vcor_rows:
                image = vcor_root / row["image_path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"test-image-placeholder")
            vcor_manifest = vcor_root / "attribute_manifest.csv"
            write_manifest(vcor_manifest, vcor_rows)
            command.extend(["--vcor-source-manifest", str(vcor_manifest)])
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_separates_vcor_gray_and_silver_from_original_folder_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [
                    base_row(source_frame_id="vcor:train/grey/a.jpg"),
                    base_row(source_frame_id="test/silver/b.jpg", split="test"),
                    base_row(source_frame_id="val/beige/c.jpg", split="validation"),
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["color"] for row in rows], ["gray", "silver", "other"])
            self.assertTrue(all(row["color_supervised"] == "true" for row in rows))
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["policy"]["existing_split_assignments_preserved_exactly"])
            self.assertFalse(report["policy"]["frozen_video_used"])

    def test_appends_only_unseen_official_vcor_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = base_row(
                image_path="crops/train/silver.jpg",
                source_frame_id="train/silver/silver.jpg",
                source_color="silver",
                sha256="2" * 64,
            )
            heldout = base_row(
                image_path="crops/test/gray.jpg",
                source_frame_id="test/grey/gray.jpg",
                source_color="grey",
                split="test",
                sha256="3" * 64,
            )
            result = self.run_builder(root, [base_row(sha256="1" * 64)], [train, heldout])
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["split"], "train")
            self.assertEqual(rows[-1]["color"], "silver")
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["appended_vcor_train_rows"], 1)
            self.assertEqual(report["append_skipped"]["heldout_source_row_not_appended"], 1)

    def test_demotes_merged_non_vcor_color_and_uses_generic_openimages_truck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [
                    base_row(
                        source_dataset="Open-Images-V7",
                        source_frame_id="abc",
                        color="silver_gray",
                        body_type="heavy_truck",
                        body_type_supervised="true",
                        official_vehicle_class="truck",
                    )
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["color"], "unknown")
            self.assertEqual(row["color_supervised"], "false")
            self.assertEqual(row["body_type"], "truck")
            self.assertEqual(row["body_type_supervised"], "true")

    def test_frozen_video_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [base_row(image_path="vcas_rtsp_demo_60s/frame.jpg")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "output.csv").exists())

    def test_v2_labels_are_offline_and_keep_production_v1_untouched(self) -> None:
        v2 = json.loads(LABELS.read_text(encoding="utf-8"))
        v1 = json.loads((LABELS.parent / "vehicle_labels.v1.json").read_text(encoding="utf-8"))
        self.assertEqual(v2["deployment_status"], "offline_candidate_only")
        self.assertIn("gray", v2["colors"])
        self.assertIn("silver", v2["colors"])
        self.assertIn("truck", v2["body_types"])
        self.assertIn("silver_gray", v1["colors"])
        self.assertNotIn("gray", v1["colors"])


if __name__ == "__main__":
    unittest.main()
