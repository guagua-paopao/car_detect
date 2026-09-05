from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from download_figshare_ranges import validate_probe  # noqa: E402


class FigshareRangeDownloadTests(unittest.TestCase):
    def test_accepts_multipart_etag_but_pins_size(self) -> None:
        result = validate_probe("bytes 0-0/10738701490", 10738701490, '"e30acefe552f6c0114778cd10690b984-11"')
        self.assertEqual(result["observed_size"], 10738701490)
        self.assertEqual(result["etag"], "e30acefe552f6c0114778cd10690b984-11")

    def test_rejects_size_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            validate_probe("bytes 0-0/9", 10, '"0123456789abcdef0123456789abcdef"')


if __name__ == "__main__":
    unittest.main()
