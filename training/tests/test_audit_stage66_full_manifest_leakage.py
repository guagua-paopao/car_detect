from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage66_full_manifest_leakage.py"


class Stage66FullLeakageAuditTest(unittest.TestCase):
    def run_audit(self, rows: list[dict[str, str]]) -> tuple[subprocess.CompletedProcess, dict]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        manifest = root / "manifest.csv"
        report = root / "report.json"
        fields = ["image_path", "split", "dhash64", "crop_sha256", "track_group"]
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--output", str(report)],
            capture_output=True, text=True, check=False,
        )
        return result, json.loads(report.read_text(encoding="utf-8"))

    def test_passes_when_all_cross_split_signals_are_disjoint(self) -> None:
        result, report = self.run_audit([
            {"image_path": "a", "split": "train", "dhash64": "ffffffffffffffff", "crop_sha256": "a" * 64, "track_group": "train-a"},
            {"image_path": "b", "split": "validation", "dhash64": "0000000000000000", "crop_sha256": "b" * 64, "track_group": "val-b"},
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "pass")

    def test_fails_on_perceptual_cross_split_overlap(self) -> None:
        result, report = self.run_audit([
            {"image_path": "a", "split": "train", "dhash64": "0000000000000001", "crop_sha256": "a" * 64, "track_group": "train-a"},
            {"image_path": "b", "split": "test", "dhash64": "0000000000000000", "crop_sha256": "b" * 64, "track_group": "test-b"},
        ])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["dhash"]["cross_split_near_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
