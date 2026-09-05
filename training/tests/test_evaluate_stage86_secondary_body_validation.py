from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_stage86_secondary_body_validation.py"
SPEC = importlib.util.spec_from_file_location("stage86_secondary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage86SecondaryTests(unittest.TestCase):
    def test_thresholds_must_come_from_passing_primary_validation(self) -> None:
        report = {
            "status": "complete_validation_only",
            "threshold_selection": {
                "source": "VFG-7 validation only",
                "body_exact": {"threshold": 0.8},
                "body_family": {"threshold": 0.75},
                "baseline_body": {"threshold": 0.7},
            },
            "shared_validation_gates": {"all_pass": True},
        }
        self.assertEqual(MODULE.extract_thresholds(report)["candidate_body"], 0.8)
        report["threshold_selection"]["source"] = "test"
        with self.assertRaisesRegex(RuntimeError, "threshold source"):
            MODULE.extract_thresholds(report)

    def test_secondary_gates_cover_hard_and_weather_views(self) -> None:
        hard = {
            "candidate": {
                "body_exact": {"precision": 0.94, "coverage": 0.50},
                "complex_body_exact": {"precision": 0.95},
            },
            "comparison": {"complex_coverage_gain": 0.16},
        }
        weather = {"candidate": {"body_exact": {"precision": 0.94, "coverage": 0.30}}}
        self.assertTrue(all(MODULE.secondary_gates(hard, weather).values()))
        weather["candidate"]["body_exact"]["precision"] = 0.92
        self.assertFalse(MODULE.secondary_gates(hard, weather)["weather_body_precision"])


if __name__ == "__main__":
    unittest.main()
