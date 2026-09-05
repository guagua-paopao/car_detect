from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "build_stage72_dvm_fine_color_manifest.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(**updates: str) -> dict[str, str]:
    value = {
        "image_path": "crops/example.jpg",
        "split": "train",
        "body_type": "unknown",
        "body_type_supervised": "false",
        "color": "silver_gray",
        "color_supervised": "true",
        "source_dataset": "DVM-CAR-2.0",
        "source_color": "Grey",
        "source_license": "CC-BY-NC-4.0;research-only;non-commercial",
        "source_group": "advert-1",
        "sha256": "1" * 64,
    }
    value.update(updates)
    return value


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Stage72DvmFineColorManifestTest(unittest.TestCase):
    def run_builder(self, root: Path, rows: list[dict[str, str]]) -> subprocess.CompletedProcess[str]:
        source = root / "dvm.csv"
        write_manifest(source, rows)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dvm-manifest",
                str(source),
                "--expected-dvm-sha256",
                sha256(source),
                "--output-manifest",
                str(root / "output.csv"),
                "--output-report",
                str(root / "report.json"),
                "--minimum-train-rows",
                "4",
                "--minimum-gray-rows",
                "1",
                "--minimum-silver-rows",
                "1",
                "--minimum-yellow-rows",
                "1",
                "--minimum-brown-rows",
                "1",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_recovers_exact_requested_colors_and_routes_related_paints_to_other(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                row(source_color="Grey", color="silver_gray", source_group="g1", sha256="1" * 64),
                row(source_color="Silver", color="silver_gray", source_group="g2", sha256="2" * 64),
                row(source_color="Yellow", color="yellow_orange", source_group="g3", sha256="3" * 64),
                row(source_color="Brown", color="brown_beige", source_group="g4", sha256="4" * 64),
                row(source_color="Orange", color="yellow_orange", source_group="g5", sha256="5" * 64),
                row(
                    split="validation",
                    source_color="Grey",
                    color="silver_gray",
                    source_group="heldout",
                    sha256="6" * 64,
                ),
            ]
            result = self.run_builder(root, rows)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output.csv").open(encoding="utf-8", newline="") as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual([item["color"] for item in output], ["gray", "silver", "yellow", "brown", "other"])
            self.assertTrue(all(item["body_type"] == "unknown" for item in output))
            self.assertTrue(all(item["taxonomy_v2_eligibility"] == "research-only_non-deployable" for item in output))
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["counters"]["heldout_validation_rows_excluded"], 1)
            self.assertFalse(report["policy"]["pixel_heuristic_used"])

    def test_rejects_conflict_between_source_metadata_and_v1_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [row(source_color="Silver", color="red")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "output.csv").exists())
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(any("conflicts with v1 color" in item for item in report["failures"]))

    def test_rejects_group_leakage_across_train_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [
                    row(source_group="same-advert", sha256="1" * 64),
                    row(split="validation", source_group="same-advert", sha256="2" * 64),
                ],
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["integrity"]["train_validation_group_leaks"], 1)

    def test_rejects_test_and_frozen_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(
                root,
                [
                    row(split="test", source_group="test-advert", sha256="1" * 64),
                    row(image_path="vcas_rtsp_demo_60s/frame.jpg", source_group="frozen", sha256="2" * 64),
                ],
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["integrity"]["test_rows"], 1)
            self.assertGreaterEqual(report["integrity"]["frozen_markers"], 1)


if __name__ == "__main__":
    unittest.main()
