from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage66_manifest_strata.py"
SPEC = importlib.util.spec_from_file_location("stage66_strata", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage66ManifestStrataTest(unittest.TestCase):
    def test_unknown_and_partial_truth_are_counted_separately(self) -> None:
        rows = [
            {"sha256": "a", "track_group": "g1", "body_type": "unknown", "body_type_supervised": "false", "color": "unknown", "color_supervised": "false", "coarse_body_family": "car", "night": "true", "source_dataset": "x", "source_license": "CC-BY-4.0"},
            {"sha256": "b", "track_group": "g2", "body_type": "bus", "body_type_supervised": "true", "color": "red", "color_supervised": "true", "coarse_body_family": "", "source_dataset": "x", "source_license": "CC-BY-4.0"},
        ]
        summary = MODULE.summarize(rows)
        self.assertEqual(summary["effective_supervision_rows"], 2)
        self.assertEqual(summary["exact_body_supervised_rows"], 1)
        self.assertEqual(summary["exact_color_supervised_rows"], 1)
        self.assertEqual(summary["coarse_body_family_rows"], {"car": 1})
        self.assertEqual(summary["scene"]["night"], 1)


if __name__ == "__main__":
    unittest.main()
