from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = Path(os.environ.get(
    "PAIR_VALIDATION_SCRIPT",
    Path(__file__).resolve().parents[1] / "scripts" / "run_stage70_pair_validation_after_training.py",
))
SPEC = importlib.util.spec_from_file_location("stage70_pair_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class Stage70PairValidationTest(unittest.TestCase):
    def test_threshold_selection_maximizes_passing_coverage(self) -> None:
        sweep = {
            "0.5": {"high_confidence_precision": 0.92, "high_confidence_coverage": 0.8, "high_confidence_selected": 8},
            "0.7": {"high_confidence_precision": 0.94, "high_confidence_coverage": 0.6, "high_confidence_selected": 6},
            "0.8": {"high_confidence_precision": 0.96, "high_confidence_coverage": 0.4, "high_confidence_selected": 4},
        }
        selected = MODULE.select_threshold(sweep)
        self.assertEqual(selected["threshold"], 0.7)
        self.assertTrue(selected["precision_gate_pass"])

    def test_pair_gates_never_use_opposite_specialist_head(self) -> None:
        production = {
            "track_validation": {
                "body_fusion5_final": {"coverage": 0.20},
                "color_fusion5_stability": {"abstention_rate": 0.50},
            }
        }
        body = {
            "selected_thresholds": {"body_type": {"precision_gate_pass": True, "coverage": 0.50}},
            "track_validation": {
                "body_fusion5_final": {"precision": 0.94, "coverage": 0.50},
                "body_fusion5_stability": {"weighted_stability_rate": 0.96},
            },
        }
        color = {
            "selected_thresholds": {"color": {"precision_gate_pass": True, "coverage": 0.30}},
            "track_validation": {
                "color_fusion5_final": {"precision": 0.95, "coverage": 0.30},
                "color_fusion5_stability": {"weighted_stability_rate": 0.97, "abstention_rate": 0.35},
            },
        }
        gates, comparison = MODULE.vfg_pair_gates(body, color, production)
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(comparison["body_fusion5_final_coverage_gain_percentage_points"], 30.0)
        self.assertAlmostEqual(comparison["color_fusion5_unknown_relative_reduction"], 0.3)

    def test_training_contract_accepts_two_teacher_specialists(self) -> None:
        training = {
            "matrix_sha256": "abc",
            "completed": [{"specialist": "color"}, {"specialist": "body"}],
            "failed": [],
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        }
        MODULE.validate_training_contract(training, "ABC", 2)

    def test_training_contract_rejects_incomplete_teacher_matrix(self) -> None:
        training = {
            "matrix_sha256": "abc",
            "completed": [{"specialist": "color"}],
            "failed": [],
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        }
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            MODULE.validate_training_contract(training, "abc", 2)


if __name__ == "__main__":
    unittest.main()
