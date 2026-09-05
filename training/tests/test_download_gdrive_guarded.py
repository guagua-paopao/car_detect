from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_gdrive_guarded.py"
SPEC = importlib.util.spec_from_file_location("guarded_gdrive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GuardedGDriveDownloadTest(unittest.TestCase):
    def test_accepts_output_inside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "nested" / "archive.zip"
            self.assertEqual(MODULE.ensure_within(output, root), output.resolve())

    def test_rejects_output_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.zip"
            with self.assertRaisesRegex(ValueError, "escapes allowed root"):
                MODULE.ensure_within(outside, root)

    def test_size_guard_stops_unbounded_download(self) -> None:
        reason = MODULE.guard_reason(output_bytes=9, free_bytes=100, max_bytes=8, min_free_bytes=10)
        self.assertEqual(reason, "max_bytes_exceeded:9>8")

    def test_free_space_guard_stops_before_disk_exhaustion(self) -> None:
        reason = MODULE.guard_reason(output_bytes=7, free_bytes=9, max_bytes=8, min_free_bytes=10)
        self.assertEqual(reason, "min_free_bytes_breached:9<10")

    def test_temporary_file_consumption_is_included_in_size_guard(self) -> None:
        observed = MODULE.effective_download_bytes(
            output_bytes=0,
            initial_free_bytes=100,
            current_free_bytes=91,
        )
        self.assertEqual(observed, 9)
        self.assertEqual(
            MODULE.guard_reason(output_bytes=observed, free_bytes=91, max_bytes=8, min_free_bytes=10),
            "max_bytes_exceeded:9>8",
        )

    def test_existing_resume_part_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "archive.zip"
            (root / "archive.zipabc.part").write_bytes(b"12345")
            self.assertEqual(MODULE.temporary_download_bytes(output), 5)


if __name__ == "__main__":
    unittest.main()
