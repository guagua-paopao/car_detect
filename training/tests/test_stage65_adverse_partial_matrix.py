from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
LOCAL_MATRIX = TRAINING / "artifacts" / "attribute-domain-v2" / "stage65-adverse-partial-matrix.json"
REMOTE_MATRIX = TRAINING.parents[1] / "runs" / "attributes" / "plans" / "stage65-adverse-partial-matrix.json"
MATRIX = LOCAL_MATRIX if LOCAL_MATRIX.is_file() else REMOTE_MATRIX
RUNNER = TRAINING / "scripts" / "run_stage65_adverse_partial_matrix.py"
SPEC = importlib.util.spec_from_file_location("stage65_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage65AdversePartialMatrixTest(unittest.TestCase):
    def test_matrix_preserves_partial_truth_and_release_isolation(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        self.assertEqual(matrix["status"], "prepared_validation_only")
        self.assertTrue(matrix["common"]["skip_test"])
        self.assertEqual(matrix["common"]["body_hierarchy"], "truck_family")
        self.assertGreater(matrix["common"]["coarse_car_loss_weight"], 0)
        self.assertTrue(matrix["execution_policy"]["deployment_paused_by_user"])
        self.assertFalse(matrix["execution_policy"]["frozen_video_used"])
        self.assertTrue(matrix["execution_policy"]["exact_body_gate_not_replaced_by_family_metric"])
        self.assertIn("leakage_report", matrix)

    def test_candidate_sampling_parameters_can_meet_required_quotas(self) -> None:
        rows = [
            {"night": "true", "small_target": "true", "occluded": "true", "adverse_supervised": "true"},
            {"night": "false", "small_target": "false", "occluded": "false", "adverse_supervised": "false"},
        ]
        quotas = MODULE.expected_sampler_quotas(rows, {
            "night_sample_weight": 4.0, "occlusion_sample_weight": 2.0,
            "small_sample_weight": 1.0, "color_sample_weight": 0.0, "hard_sample_weight": 0.0,
        })
        self.assertGreater(quotas["night"], 0.8)
        self.assertEqual(quotas["night"], quotas["small"])
        self.assertEqual(quotas["night"], quotas["occluded_or_truncated"])

    def test_metadata_is_not_forwarded_to_training(self) -> None:
        args = MODULE.command_arguments({"candidate_id": "x", "purpose": "p", "coarse_car_loss_weight": 0.2, "skip_test": True})
        self.assertNotIn("--candidate-id", args)
        self.assertNotIn("--purpose", args)
        self.assertIn("--coarse-car-loss-weight", args)
        self.assertIn("--skip-test", args)


if __name__ == "__main__":
    unittest.main()
