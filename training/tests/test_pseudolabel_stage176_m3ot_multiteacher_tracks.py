from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pseudolabel_stage176_m3ot_multiteacher_tracks.py"
SPEC = importlib.util.spec_from_file_location("stage176", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage176PolicyTests(unittest.TestCase):
    def test_maps_legacy_taxonomy_by_exact_name(self) -> None:
        canonical = ["sedan", "truck", "unknown"]
        self.assertEqual(MODULE.canonical_label(["sedan", "unknown"], 0, canonical), "sedan")
        self.assertEqual(MODULE.canonical_label(["sedan", "unknown"], 5, canonical), "unknown")

    def test_track_requires_exact_label_consistency(self) -> None:
        self.assertIsNone(MODULE.qualify_track(["sedan", "suv", "sedan"], 4, 3, 0.5))
        self.assertEqual(MODULE.qualify_track(["sedan", "sedan", "sedan"], 4, 3, 0.5), "sedan")

    def test_track_requires_minimum_strong_fraction(self) -> None:
        self.assertIsNone(MODULE.qualify_track(["sedan", "sedan", "sedan"], 20, 3, 0.25))


if __name__ == "__main__":
    unittest.main()
