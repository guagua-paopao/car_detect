from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stage71_final_test_after_student_validation.py"
)
SPEC = importlib.util.spec_from_file_location("stage71_final_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def policy() -> dict:
    return {
        "validation_only": True,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def pair(pair_id: str, body_coverage: float, color_coverage: float) -> dict:
    return {
        "pair_id": pair_id,
        "screen_status": "pass",
        "hard": {
            "body_type": {
                "threshold": 0.8,
                "precision": 0.94,
                "coverage": body_coverage,
            },
            "color": {
                "threshold": 0.7,
                "precision": 0.94,
                "coverage": color_coverage,
            },
        },
        "vfg7": {
            "body": {"track": {"body_fusion5_final": {"coverage": 0.5}}},
            "color": {"track": {"color_fusion5_final": {"coverage": 0.4}}},
        },
    }


def write_state(tmp_path: Path, report: dict) -> dict:
    report_path = tmp_path / "pair-validation-screen-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "report_sha256": MODULE.sha256(report_path),
    }


class Stage71FinalTestGateTests(unittest.TestCase):
    def test_validation_report_rejects_no_passing_pair_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = write_state(
                Path(directory),
                {"policy": policy(), "passing_pairs": [], "pairs": []},
            )
            with self.assertRaisesRegex(RuntimeError, "test remains locked"):
                MODULE.validate_validation_report(state)

    def test_validation_report_rejects_policy_violation(self) -> None:
        bad_policy = policy()
        bad_policy["test_accessed"] = True
        with tempfile.TemporaryDirectory() as directory:
            state = write_state(
                Path(directory),
                {
                    "policy": bad_policy,
                    "passing_pairs": ["a"],
                    "pairs": [pair("a", 0.5, 0.3)],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "test_accessed"):
                MODULE.validate_validation_report(state)

    def test_pair_selection_uses_validation_coverage_only(self) -> None:
        report = {
            "passing_pairs": ["lower", "higher"],
            "pairs": [
                pair("lower", 0.46, 0.26),
                pair("higher", 0.55, 0.40),
            ],
        }
        self.assertEqual(MODULE.selected_pair(report)["pair_id"], "higher")

    def test_unknown_reduction_is_relative(self) -> None:
        # Baseline unknown=0.50, candidate unknown=0.35 => 30% reduction.
        self.assertAlmostEqual(
            MODULE.relative_unknown_reduction(0.50, 0.65),
            0.30,
        )


if __name__ == "__main__":
    unittest.main()
