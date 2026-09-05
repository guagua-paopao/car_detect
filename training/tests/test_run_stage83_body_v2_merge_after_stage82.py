from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage83_body_v2_merge_after_stage82.py"
SPEC = importlib.util.spec_from_file_location("stage83_waiter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage83WaiterTests(unittest.TestCase):
    def test_report_validation_binds_manifest_and_zero_heldout_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "stage82.csv"
            manifest.write_text("image_path,split\na.jpg,train\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps({
                "status": "pass",
                "output": {"manifest": str(manifest), "manifest_sha256": MODULE.sha256(manifest)},
                "policy": {"official_test_image_payloads_read": 0, "validation_rows_created": 0, "test_rows_created": 0},
            }), encoding="utf-8")
            parsed, digest = MODULE.validate_stage82_report(report, manifest)
            self.assertEqual(parsed["status"], "pass")
            self.assertEqual(digest, MODULE.sha256(manifest))

    def test_report_validation_rejects_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "stage82.csv"
            manifest.write_text("image_path,split\na.jpg,train\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps({
                "status": "pass",
                "output": {"manifest": str(manifest), "manifest_sha256": MODULE.sha256(manifest)},
                "policy": {"official_test_image_payloads_read": 1, "validation_rows_created": 0, "test_rows_created": 0},
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.validate_stage82_report(report, manifest)


if __name__ == "__main__":
    unittest.main()
