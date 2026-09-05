from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "decide_stage67_uvh26_followup.py"
SPEC = importlib.util.spec_from_file_location("stage67_decision", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(name: str, *, type_pass: bool, full_pass: bool = False) -> dict:
    gates = {key: type_pass for key in MODULE.TYPE_GATES}
    vfg_gates = {key: type_pass for key in MODULE.VFG_TYPE_GATES}
    return {
        "model": name,
        "checkpoint": f"/{name}.pt",
        "checkpoint_sha256": "a" * 64,
        "screen_status": "pass" if full_pass else "fail_closed",
        "gates": gates,
        "hard": {"body_type": {"precision": 0.95, "coverage": 0.50 if type_pass else 0.40}},
        "ua_track": {"body_type": {"precision": 0.95, "coverage": 0.50 if type_pass else 0.40, "weighted_stability_rate": 0.99}},
        "vfg7": {
            "validation_gates": vfg_gates,
            "selected_thresholds": {"body_type": {"precision": 0.95, "coverage": 0.50 if type_pass else 0.40}},
        },
    }


class Stage67DecisionTest(unittest.TestCase):
    def test_runs_only_when_no_candidate_passes_type_gates(self) -> None:
        report = {"models": [candidate("production", type_pass=False), candidate("candidate-a", type_pass=False)], "passing_candidates": []}
        result = MODULE.decision_from_report(report)
        self.assertEqual(result["action"], "run_stage67_uvh26_controlled_type_followup")

    def test_skips_type_only_followup_when_type_is_not_limiting(self) -> None:
        report = {"models": [candidate("production", type_pass=False), candidate("candidate-a", type_pass=True)], "passing_candidates": []}
        result = MODULE.decision_from_report(report)
        self.assertEqual(result["action"], "skip_stage67_type_gates_met_focus_next_round_on_color")
        self.assertEqual(result["type_passing_candidates"], ["candidate-a"])

    def test_full_stage66_pass_takes_priority(self) -> None:
        report = {"models": [candidate("production", type_pass=False), candidate("candidate-a", type_pass=True, full_pass=True)], "passing_candidates": ["candidate-a"]}
        result = MODULE.decision_from_report(report)
        self.assertEqual(result["action"], "skip_stage67_stage66_full_candidate_available")

    def test_partial_vfg_type_failure_triggers_followup(self) -> None:
        item = candidate("candidate-a", type_pass=True)
        item["vfg7"]["validation_gates"]["track_body"] = False
        result = MODULE.decision_from_report({"models": [candidate("production", type_pass=False), item], "passing_candidates": []})
        self.assertEqual(result["action"], "run_stage67_uvh26_controlled_type_followup")
        self.assertFalse(result["candidate_type_summaries"][0]["type_gate_pass"])

    def test_empty_candidate_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no candidate"):
            MODULE.decision_from_report({"models": [candidate("production", type_pass=False)], "passing_candidates": []})


if __name__ == "__main__":
    unittest.main()
