from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "apply_stage71_fail_closed_v2_test_gate.py"
SPEC = importlib.util.spec_from_file_location("stage71_v2_test_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(precision: float, coverage: float, stability: float) -> dict:
    return {
        "precision": precision,
        "coverage": coverage,
        "effective_unknown_rate": 1.0 - coverage,
        "weighted_stability_rate": stability,
    }


class Stage71FailClosedV2TestGateTest(unittest.TestCase):
    def test_all_fixed_test_gates_pass(self) -> None:
        gates, comparison = MODULE.test_gates(
            metrics(0.94, 0.70, 0.97),
            metrics(0.95, 0.70, 0.98),
            metrics(0.94, 0.60, 0.96),
            metrics(0.94, 0.50, 0.96),
            metrics(0.94, 0.50, 0.96),
        )
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(comparison["body_coverage_gain_percentage_points"], 20.0)
        self.assertAlmostEqual(comparison["color_unknown_relative_reduction"], 0.4)

    def test_test_stability_failure_blocks_backend(self) -> None:
        gates, _ = MODULE.test_gates(
            metrics(0.99, 0.70, 0.90),
            metrics(0.99, 0.70, 0.90),
            metrics(0.99, 0.60, 0.90),
            metrics(0.94, 0.50, 0.96),
            metrics(0.94, 0.50, 0.96),
        )
        self.assertFalse(gates["v2_test_vfg_body_stability_gte_0_95"])
        self.assertFalse(gates["v2_test_vfg_color_stability_gte_0_95"])
        self.assertFalse(gates["v2_test_ua_body_stability_gte_0_95"])

    def test_missing_fixed_window_is_fail_closed(self) -> None:
        compact = MODULE.compact_fixed({"body_type": {"fusion": {}}}, "body_type", 5)
        self.assertEqual(compact["coverage"], 0.0)
        self.assertEqual(compact["effective_unknown_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
