from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_m3ot_rgb_train.py"
SPEC = importlib.util.spec_from_file_location("m3ot_extract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M3OTExtractionTests(unittest.TestCase):
    def fixture(self, root: Path, *, audit_status: str = "pass") -> tuple[Path, Path]:
        archive = root / "M3OT.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as opened:
            for drone in ("1", "2"):
                opened.writestr(f"M3OT/Annotations/{drone}/rgb/train_cocoformat.json", b"{}")
                opened.writestr(f"M3OT/{drone}/rgb/train/{drone}-01/img1/000001.PNG", f"rgb-{drone}".encode())
                opened.writestr(f"M3OT/{drone}/rgb/val/{drone}-08/img1/000001.PNG", b"validation-must-not-extract")
                opened.writestr(f"M3OT/{drone}/ir/train/{drone}-01T/img1/000001.PNG", b"ir-must-not-extract")
        audit = root / "audit.json"
        audit.write_text(json.dumps({"status": audit_status, "archive_sha256": MODULE.sha256_file(archive), "validation_payload_opened": False, "test_payload_opened": False}), encoding="utf-8")
        return archive, audit

    def test_extracts_only_rgb_train_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, audit = self.fixture(root)
            output = root / "output"
            report = MODULE.extract_rgb_train(archive, audit, output, expected_audit_sha256=MODULE.sha256_file(audit), expected_images=2, minimum_free_bytes=0, maximum_output_bytes=1024 * 1024)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["rgb_train_images"], 2)
            self.assertTrue((output / "1/rgb/train/1-01/img1/000001.PNG").is_file())
            self.assertFalse((output / "1/rgb/val/1-08/img1/000001.PNG").exists())
            self.assertFalse((output / "1/ir/train/1-01T/img1/000001.PNG").exists())

    def test_rejects_failed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, audit = self.fixture(root, audit_status="fail")
            with self.assertRaises(ValueError):
                MODULE.extract_rgb_train(archive, audit, root / "output", expected_audit_sha256=MODULE.sha256_file(audit), expected_images=2, minimum_free_bytes=0, maximum_output_bytes=1024 * 1024)

    def test_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, audit = self.fixture(root)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.extract_rgb_train(archive, audit, output, expected_audit_sha256=MODULE.sha256_file(audit), expected_images=2, minimum_free_bytes=0, maximum_output_bytes=1024 * 1024)


if __name__ == "__main__":
    unittest.main()
