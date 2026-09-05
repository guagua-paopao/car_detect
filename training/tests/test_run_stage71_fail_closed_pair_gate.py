from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "run_stage71_fail_closed_pair_gate.py"
SPEC = importlib.util.spec_from_file_location("stage71_track_gate_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(
    precision: float,
    coverage: float,
    stability: float,
    unknown: float | None = None,
) -> dict:
    return {
        "precision": precision,
        "coverage": coverage,
        "weighted_stability_rate": stability,
        "effective_unknown_rate": 1.0 - coverage if unknown is None else unknown,
    }


class Stage71FailClosedPairGateTest(unittest.TestCase):
    def test_all_required_gates_pass(self) -> None:
        gates, comparison = MODULE.pair_gates(
            metrics(0.94, 0.70, 0.97),
            metrics(0.95, 0.70, 0.98, 0.30),
            metrics(0.94, 0.60, 0.96),
            metrics(0.94, 0.50, 0.96),
            metrics(0.94, 0.50, 0.96, 0.50),
        )
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(comparison["body_coverage_gain_percentage_points"], 20.0)
        self.assertAlmostEqual(comparison["color_unknown_relative_reduction"], 0.4)

    def test_conflict_abstention_coverage_regression_fails_closed(self) -> None:
        gates, _ = MODULE.pair_gates(
            metrics(0.99, 0.40, 0.99),
            metrics(0.99, 0.20, 0.99, 0.80),
            metrics(0.99, 0.40, 0.99),
            metrics(0.94, 0.50, 0.96),
            metrics(0.94, 0.50, 0.96, 0.50),
        )
        self.assertFalse(gates["v2_vfg_body_coverage_gte_0_45"])
        self.assertFalse(gates["v2_vfg_color_coverage_gte_0_25"])
        self.assertFalse(gates["v2_color_unknown_reduction_gte_20pct"])

    def test_no_selection_is_explicitly_non_passing(self) -> None:
        compact = MODULE.compact_selected({"body_type": {"selected": None}}, "body_type")
        self.assertEqual(compact["selection_status"], "no_selection")
        self.assertEqual(compact["coverage"], 0.0)
        self.assertEqual(compact["effective_unknown_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
