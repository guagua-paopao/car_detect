from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_stage62_stage63_validation.py"
SPEC = importlib.util.spec_from_file_location("combined_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(*, passed: bool, gates: int, body_gain: float, color_gain: float) -> dict:
    gate_map = {f"gate_{index}": index < gates for index in range(4)}
    return {
        "screen_status": "pass" if passed else "fail_closed",
        "gates": gate_map,
        "comparison_to_production": {
            "hard_body_coverage_gain_percentage_points": body_gain,
            "hard_color_unknown_relative_reduction": color_gain,
        },
        "hard": {"body_type": {"coverage": 0.5}, "color": {"coverage": 0.4}},
    }


class CombinedValidationRankingTest(unittest.TestCase):
    def test_full_gate_pass_always_ranks_before_partial_candidate(self) -> None:
        passed = candidate(passed=True, gates=4, body_gain=15.0, color_gain=0.2)
        failed = candidate(passed=False, gates=4, body_gain=30.0, color_gain=0.5)
        self.assertGreater(MODULE.rank_key(passed), MODULE.rank_key(failed))

    def test_failure_ranking_uses_gate_count_before_metric_gain(self) -> None:
        more_gates = candidate(passed=False, gates=3, body_gain=5.0, color_gain=0.1)
        more_gain = candidate(passed=False, gates=2, body_gain=40.0, color_gain=0.8)
        self.assertGreater(MODULE.rank_key(more_gates), MODULE.rank_key(more_gain))


if __name__ == "__main__":
    unittest.main()
