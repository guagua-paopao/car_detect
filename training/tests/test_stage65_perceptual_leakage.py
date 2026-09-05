from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "audit_stage65_perceptual_leakage.py"
SPEC = importlib.util.spec_from_file_location("stage65_leakage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage65PerceptualLeakageTest(unittest.TestCase):
    def test_exact_multi_index_finds_distance_four(self) -> None:
        base = 0x1234567890ABCDEF
        changed = base ^ ((1 << 1) | (1 << 17) | (1 << 35) | (1 << 58))
        self.assertEqual(MODULE.find_near_pairs([base], [changed]), [(0, 0, 4)])

    def test_distance_five_is_rejected(self) -> None:
        base = 0
        changed = sum(1 << bit for bit in (1, 14, 27, 40, 53))
        self.assertEqual(MODULE.find_near_pairs([base], [changed]), [])

    def test_invalid_hash_fails_closed_to_pixel_recompute(self) -> None:
        self.assertIsNone(MODULE.parse_dhash("not-a-hash"))
        self.assertEqual(MODULE.parse_dhash("000000000000000f"), 15)


if __name__ == "__main__":
    unittest.main()
