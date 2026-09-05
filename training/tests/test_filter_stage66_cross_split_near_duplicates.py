from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "filter_stage66_cross_split_near_duplicates.py"


class Stage66NearDuplicateFilterTest(unittest.TestCase):
    def test_near_train_row_is_removed_and_heldout_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "input.csv"
            output = root / "output.csv"
            removed = root / "removed.csv"
            report = root / "report.json"
            fields = ["image_path", "split", "dhash64", "crop_sha256"]
            rows = [
                {"image_path": "near.jpg", "split": "train", "dhash64": "0000000000000001", "crop_sha256": "1" * 64},
                {"image_path": "far.jpg", "split": "train", "dhash64": "ffffffffffffffff", "crop_sha256": "2" * 64},
                {"image_path": "val.jpg", "split": "validation", "dhash64": "0000000000000000", "crop_sha256": "3" * 64},
                {"image_path": "test.jpg", "split": "test", "dhash64": "0f0f0f0f0f0f0f0f", "crop_sha256": "4" * 64},
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            subprocess.run([
                sys.executable, str(SCRIPT), "--manifest", str(source),
                "--output-manifest", str(output), "--removed-manifest", str(removed),
                "--output-report", str(report),
            ], check=True, capture_output=True, text=True)
            with output.open("r", encoding="utf-8", newline="") as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual([row["image_path"] for row in result], ["far.jpg", "val.jpg", "test.jpg"])
            evidence = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "pass")
            self.assertEqual(evidence["removed_train_rows"], 1)
            self.assertEqual(evidence["residual_cross_split_near_pairs"], 0)
            self.assertEqual(evidence["heldout_rows_preserved"], 2)


if __name__ == "__main__":
    unittest.main()
