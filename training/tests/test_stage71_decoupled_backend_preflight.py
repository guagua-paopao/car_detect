from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_stage71_decoupled_onnx.py"
)
SPEC = importlib.util.spec_from_file_location("stage71_export", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def eligible_report() -> dict:
    return {
        "status": "pass_backend_eligible",
        "gates": {"a": True, "b": True},
        "policy": {
            "test_accessed": True,
            "test_used_for_selection": False,
            "one_validation_selected_pair_tested": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }


class Stage71BackendPreflightTests(unittest.TestCase):
    def test_accepts_fully_gated_candidate(self) -> None:
        MODULE.validate_final_report(eligible_report())

    def test_rejects_failed_metric_gate(self) -> None:
        report = eligible_report()
        report["gates"]["b"] = False
        with self.assertRaisesRegex(RuntimeError, "failed gate"):
            MODULE.validate_final_report(report)

    def test_rejects_test_selection_leak(self) -> None:
        report = eligible_report()
        report["policy"]["test_used_for_selection"] = True
        with self.assertRaisesRegex(RuntimeError, "test_used_for_selection"):
            MODULE.validate_final_report(report)

    def test_rejects_frozen_video_use(self) -> None:
        report = eligible_report()
        report["policy"]["frozen_video_used"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen_video_used"):
            MODULE.validate_final_report(report)


if __name__ == "__main__":
    unittest.main()
