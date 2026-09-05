from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_http_range_batches.py"
SPEC = importlib.util.spec_from_file_location("range_downloader", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HttpRangeBatchTest(unittest.TestCase):
    def test_ranges_cover_every_byte_once(self) -> None:
        plan = MODULE.ranges(10, 4)
        self.assertEqual(plan, [(0, 0, 3), (1, 4, 7), (2, 8, 9)])
        self.assertEqual(sum(end - start + 1 for _, start, end in plan), 10)

    def test_invalid_sizes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.ranges(0, 4)
        with self.assertRaises(ValueError):
            MODULE.ranges(10, 0)


if __name__ == "__main__":
    unittest.main()
