from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(os.environ.get(
    "SPECIALIST_MATRIX_SCRIPT",
    Path(__file__).resolve().parents[1] / "scripts" / "run_stage70_decoupled_matrix.py",
))
SPEC = importlib.util.spec_from_file_location("stage70_matrix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class Stage70MatrixTest(unittest.TestCase):
    def test_command_omits_metadata_and_preserves_skip_test(self) -> None:
        arguments = MODULE.command_arguments({
            "candidate_id": "C0", "specialist": "body", "purpose": "test",
            "skip_test": True, "body_loss_weight": 1.0, "color_loss_weight": 0.0,
        })
        self.assertIn("--skip-test", arguments)
        self.assertNotIn("--candidate-id", arguments)
        self.assertNotIn("--specialist", arguments)

    def test_quota_and_head_filtering(self) -> None:
        rows = [
            {"split": "train", "review_status": "approved", "body_type": "sedan",
             "body_type_supervised": "true", "color": "unknown", "color_supervised": "false",
             "night": "true", "vehicle_size": "small"},
            {"split": "train", "review_status": "approved", "body_type": "van",
             "body_type_supervised": "true", "color": "unknown", "color_supervised": "false",
             "night": "false"},
        ]
        usable = MODULE.usable_train_rows(rows, "body")
        quota = MODULE.expected_supervised_quota(usable, {
            "night_sample_weight": 4.0, "small_sample_weight": 0.0,
            "occlusion_sample_weight": 1.0,
        })
        self.assertEqual(quota["rows"], 2)
        self.assertAlmostEqual(quota["night"], 0.8)
        raw = [
            {"split": "train", "body_type": "unknown", "color": "unknown",
             "body_type_supervised": "false", "color_supervised": "false", "night": "true"},
            {"split": "train", "body_type": "unknown", "color": "unknown",
             "body_type_supervised": "false", "color_supervised": "false", "night": "false"},
        ]
        self.assertAlmostEqual(MODULE.expected_unlabeled_night(raw, 4.0), 0.8)

    def test_stage71_body_teacher_must_match_passing_pair(self) -> None:
        pair = {
            "body_checkpoint": "/teachers/body.pt",
            "body_checkpoint_sha256": "a" * 64,
            "color_checkpoint": "/teachers/color.pt",
            "color_checkpoint_sha256": "b" * 64,
        }
        MODULE.validate_teacher_binding({
            "specialist": "body",
            "body_teacher_checkpoint": "/teachers/body.pt",
            "body_teacher_checkpoint_sha256": "A" * 64,
        }, pair)

    def test_stage71_rejects_cross_head_teacher_binding(self) -> None:
        pair = {
            "body_checkpoint": "/teachers/body.pt",
            "body_checkpoint_sha256": "a" * 64,
            "color_checkpoint": "/teachers/color.pt",
            "color_checkpoint_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "color teacher"):
            MODULE.validate_teacher_binding({
                "specialist": "body",
                "body_teacher_checkpoint": "/teachers/body.pt",
                "body_teacher_checkpoint_sha256": "a" * 64,
                "color_teacher_checkpoint": "/teachers/color.pt",
            }, pair)

    def test_stage71_teacher_evidence_requires_a_fully_passing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.json"
            training = root / "training.json"
            manifests = root / "manifests.json"
            validation.write_text(json.dumps({
                "status": "pass_pairs_available",
                "passing_pairs": ["pair"],
                "pairs": [{"pair_id": "pair", "screen_status": "pass"}],
                "policy": {
                    "validation_only": True,
                    "test_accessed": False,
                    "frozen_video_used": False,
                    "production_model_modified": False,
                    "backend_gates_run": False,
                    "deployment_performed": False,
                },
            }), encoding="utf-8")
            training.write_text(json.dumps({
                "status": "complete_validation_only",
                "completed": [{}, {}],
                "failed": [],
                "test_accessed": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "deployment_performed": False,
            }), encoding="utf-8")
            manifests.write_text(json.dumps({
                "status": "pass",
                "eligibility": "research-only_non-deployable",
                "outputs": {"color_teacher": {"splits": {"train": 118338}}},
                "post_filter_cross_split_near_leaks": [],
                "post_filter_group_leaks": [],
            }), encoding="utf-8")
            pair = MODULE.validate_stage71_teacher_evidence({
                "teacher_validation_report": validation,
                "teacher_training_state": training,
                "teacher_manifest_report": manifests,
            }, {"teacher_pair_id": "pair"})
            self.assertEqual(pair["pair_id"], "pair")


if __name__ == "__main__":
    unittest.main()
