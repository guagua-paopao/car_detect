from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_stage173_projected_scene_quotas import additional_rows_needed  # noqa: E402


class Stage173QuotaProjectionTests(unittest.TestCase):
    def test_no_rows_needed_when_target_met(self) -> None:
        self.assertEqual(additional_rows_needed(100, 30, 0.30), 0)

    def test_rows_needed_solves_expanding_denominator(self) -> None:
        needed = additional_rows_needed(100, 10, 0.30)
        self.assertEqual(needed, 29)
        self.assertGreaterEqual((10 + needed) / (100 + needed), 0.30)
        self.assertLess((10 + needed - 1) / (100 + needed - 1), 0.30)


if __name__ == "__main__":
    unittest.main()
