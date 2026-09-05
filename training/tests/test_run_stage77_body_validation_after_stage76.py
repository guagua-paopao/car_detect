import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stage77_body_validation_after_stage76.py"
)
SPEC = importlib.util.spec_from_file_location("stage77", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_maximum_coverage_subject_to_precision(self):
        report = {
            "threshold_sweep": {
                "body_type": {
                    "0.700": {
                        "evaluated": 100,
                        "high_confidence_selected": 80,
                        "high_confidence_precision": 0.92,
                        "high_confidence_coverage": 0.80,
                    },
                    "0.800": {
                        "evaluated": 100,
                        "high_confidence_selected": 60,
                        "high_confidence_precision": 0.94,
                        "high_confidence_coverage": 0.60,
                    },
                    "0.900": {
                        "evaluated": 100,
                        "high_confidence_selected": 40,
                        "high_confidence_precision": 0.98,
                        "high_confidence_coverage": 0.40,
                    },
                }
            }
        }
        selected = MODULE.select_body_threshold(report)
        self.assertEqual(selected["threshold"], 0.8)
        self.assertTrue(selected["precision_gate_found"])

    def test_falls_back_fail_closed_when_precision_gate_missing(self):
        report = {
            "threshold_sweep": {
                "body_type": {
                    "0.700": {
                        "evaluated": 100,
                        "high_confidence_selected": 80,
                        "high_confidence_precision": 0.90,
                        "high_confidence_coverage": 0.80,
                    },
                    "0.900": {
                        "evaluated": 100,
                        "high_confidence_selected": 20,
                        "high_confidence_precision": 0.92,
                        "high_confidence_coverage": 0.20,
                    },
                }
            }
        }
        selected = MODULE.select_body_threshold(report)
        self.assertEqual(selected["threshold"], 0.9)
        self.assertFalse(selected["precision_gate_found"])


class GateTests(unittest.TestCase):
    def test_all_body_gates_must_pass(self):
        static = {"precision": 0.94, "coverage": 0.61}
        production = {"precision": 0.94, "coverage": 0.45}
        hard = {"precision": 0.95, "coverage": 0.50}
        track = {
            "final_precision": 0.94,
            "final_coverage": 0.50,
            "weighted_stability_rate": 0.97,
        }
        decision = MODULE.body_gate_decision(static, production, hard, track)
        self.assertTrue(decision["all_gates_pass"])

    def test_rejects_coverage_gain_below_fifteen_points(self):
        static = {"precision": 0.94, "coverage": 0.59}
        production = {"precision": 0.94, "coverage": 0.45}
        hard = {"precision": 0.95, "coverage": 0.50}
        track = {
            "final_precision": 0.94,
            "final_coverage": 0.50,
            "weighted_stability_rate": 0.97,
        }
        decision = MODULE.body_gate_decision(static, production, hard, track)
        self.assertFalse(decision["all_gates_pass"])
        self.assertFalse(decision["gates"]["complex_coverage_gain"])


if __name__ == "__main__":
    unittest.main()
