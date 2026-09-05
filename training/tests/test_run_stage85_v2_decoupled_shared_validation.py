from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage85_v2_decoupled_shared_validation.py"
SPEC = importlib.util.spec_from_file_location("run_stage85_shared", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage85RunnerTests(unittest.TestCase):
    def test_dependency_actions_fail_closed(self) -> None:
        self.assertEqual(MODULE.dependency_action("running"), "wait")
        self.assertEqual(MODULE.dependency_action("waiting_for_stage76_stage80"), "wait")
        self.assertEqual(MODULE.dependency_action("complete_validation_only"), "ready")
        self.assertEqual(MODULE.dependency_action("failed_closed_training"), "fail")
        self.assertEqual(MODULE.dependency_action("rejected_validation_gate_failure"), "fail")

    def test_checkpoint_lineage_requires_matching_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"candidate")
            state = {
                "status": "complete_validation_only",
                "best_checkpoint": str(checkpoint),
                "best_checkpoint_sha256": MODULE.sha256(checkpoint),
            }
            selected, digest = MODULE.select_checkpoint(state, "candidate")
            self.assertEqual(selected, checkpoint)
            self.assertEqual(digest, MODULE.sha256(checkpoint))
            state["best_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                MODULE.select_checkpoint(state, "candidate")

    def test_report_contract_requires_complex_and_per_class_evidence(self) -> None:
        gates = {
            "body_static_precision": True,
            "body_static_coverage": True,
            "body_complex_coverage_gain": True,
            "color_static_precision": True,
            "color_static_coverage": True,
            "color_complex_unknown_reduction": True,
            "body_track_precision": True,
            "body_track_coverage": True,
            "color_track_precision": True,
            "color_track_coverage": True,
            "body_track_stability": True,
            "color_track_stability": True,
        }
        report = {
            "status": "complete_validation_only",
            "policy": {
                "test_accessed": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
            },
            "inputs": {"body_checkpoint_sha256": "a", "color_checkpoint_sha256": "b"},
            "rows": {"complex_body_supervised": 1, "complex_color_supervised": 1},
            "complex_static": {},
            "per_class": {},
            "comparison": {
                "body_complex_static_coverage_gain": 0.2,
                "color_complex_static_unknown_relative_reduction": 0.3,
            },
            "shared_validation_gates": {"gates": gates},
        }
        MODULE.validate_report_contract(report, "a", "b")
        del report["comparison"]["body_complex_static_coverage_gain"]
        with self.assertRaisesRegex(RuntimeError, "complex comparison"):
            MODULE.validate_report_contract(report, "a", "b")


if __name__ == "__main__":
    unittest.main()
