from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "run_stage67_uvh26_controlled_matrix.py"
LOCAL_MATRIX = TRAINING_ROOT / "artifacts" / "attribute-domain-v2" / "stage67-uvh26-controlled-matrix.json"
MATRIX = LOCAL_MATRIX if LOCAL_MATRIX.is_file() else Path(
    "/root/autodl-tmp/vcas/runs/attributes/plans/stage67-uvh26-controlled-matrix.json"
)
SPEC = importlib.util.spec_from_file_location("stage67_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage67ControlledMatrixTest(unittest.TestCase):
    def test_group_key_uses_track_group_when_source_group_is_blank(self) -> None:
        self.assertEqual(
            MODULE.source_group_key({"source_group": "", "track_group": "UVH26:42", "source_frame_id": "frame.png"}),
            "UVH26:42",
        )

    def test_matrix_is_conditional_production_initialized_and_color_preserving(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        self.assertEqual(matrix["status"], "prepared_conditional_validation_only")
        self.assertEqual(matrix["runner_script_sha256"], MODULE.sha256(SCRIPT).upper())
        self.assertTrue(matrix["execution_policy"]["requires_stage66_type_gap_decision"])
        self.assertFalse(matrix["execution_policy"]["test_accessed"])
        self.assertFalse(matrix["execution_policy"]["frozen_video_used"])
        self.assertEqual({item["input_size"] for item in matrix["candidates"]}, {224, 256})
        for item in matrix["candidates"]:
            self.assertEqual(item["architecture"], "mobilenet_v3_large")
            self.assertEqual(item["init_checkpoint"], item["teacher_checkpoint"])
            self.assertEqual(item["init_checkpoint_sha256"], item["teacher_checkpoint_sha256"])
            self.assertGreater(item["distill_weight"], 0)

    def test_dependency_hash_metadata_is_not_forwarded(self) -> None:
        args = MODULE.command_arguments({
            "candidate_id": "u0", "purpose": "test", "init_checkpoint": "/p.pt",
            "init_checkpoint_sha256": "a" * 64, "teacher_checkpoint_sha256": "a" * 64,
            "skip_test": True,
        })
        self.assertIn("--init-checkpoint", args)
        self.assertIn("--skip-test", args)
        self.assertNotIn("--init-checkpoint-sha256", args)

    def test_safe_policy_requires_deployment_pause(self) -> None:
        policy = {key: False for key in MODULE.SAFE_FALSE_KEYS}
        policy["test_accessed"] = False
        policy["deployment_paused_by_user"] = True
        MODULE.require_safe_policy(policy, "fixture")
        policy["frozen_video_used"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen_video_used"):
            MODULE.require_safe_policy(policy, "fixture")

    def test_safe_policy_accepts_explicit_long_form_test_isolation(self) -> None:
        policy = {key: False for key in MODULE.SAFE_FALSE_KEYS}
        policy["test_labels_or_predictions_accessed"] = False
        policy["deployment_paused_by_user"] = True
        MODULE.require_safe_policy(policy, "fixture")

    def test_decision_must_explicitly_require_stage67(self) -> None:
        # Hash comparison is reached only after action and policy validation; a
        # skipped decision must fail before any report file is accessed.
        policy = {key: False for key in MODULE.SAFE_FALSE_KEYS}
        policy["test_accessed"] = False
        policy["deployment_paused_by_user"] = True
        decision = {"status": "pass", "action": "skip_stage67_type_gates_met_focus_next_round_on_color", "policy": policy}
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            MODULE.validate_decision(decision, Path("/does/not/matter"))


if __name__ == "__main__":
    unittest.main()
