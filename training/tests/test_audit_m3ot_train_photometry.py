from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_m3ot_train_photometry.py"
SPEC = importlib.util.spec_from_file_location("m3ot_photometry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M3OTPhotometryTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        extraction = root / "extraction"
        levels = {"01": 20, "02": 100, "04": 220}
        for drone in ("1", "2"):
            for suffix, level in levels.items():
                image_root = extraction / drone / "rgb" / "train" / f"{drone}-{suffix}" / "img1"
                image_root.mkdir(parents=True)
                Image.new("RGB", (640, 512), (level, level, level)).save(image_root / "000001.PNG")
        report = root / "extraction.json"
        report.write_text(json.dumps({"status": "pass", "output_root": str(extraction), "validation_payload_opened": False, "test_payload_opened": False}), encoding="utf-8")
        return extraction, report

    def test_clusters_paired_day_dusk_night(self) -> None:
        labels, centers = MODULE.cluster_conditions({"01": 20.0, "02": 100.0, "04": 220.0})
        self.assertEqual(labels, {"01": "night", "02": "dusk", "04": "day"})
        self.assertLess(centers["night"], centers["dusk"])
        self.assertLess(centers["dusk"], centers["day"])

    def test_audit_decodes_train_only_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction, report = self.fixture(root)
            result = MODULE.audit_photometry(extraction, report, expected_extraction_report_sha256=MODULE.sha256_file(report), expected_images=6)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["decoded_rgb_train_images"], 6)
            self.assertFalse(result["validation_payload_opened"])
            self.assertTrue(any("expected 14" in failure for failure in result["failures"]))

    def test_rejects_report_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction, report = self.fixture(root)
            with self.assertRaises(ValueError):
                MODULE.audit_photometry(extraction, report, expected_extraction_report_sha256="0" * 64, expected_images=6)


if __name__ == "__main__":
    unittest.main()
