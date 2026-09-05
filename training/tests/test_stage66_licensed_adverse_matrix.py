from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "run_stage66_licensed_adverse_matrix.py"
LOCAL_MATRIX = TRAINING_ROOT / "artifacts" / "attribute-domain-v2" / "stage66-licensed-adverse-architecture-matrix.json"
MATRIX = LOCAL_MATRIX if LOCAL_MATRIX.is_file() else Path(
    "/root/autodl-tmp/vcas/runs/attributes/plans/stage66-licensed-adverse-architecture-matrix.json"
)
SPEC = importlib.util.spec_from_file_location("stage66_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage66LicensedAdverseMatrixTest(unittest.TestCase):
    def test_matrix_preserves_production_license_and_isolation_policy(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        policy = matrix["execution_policy"]
        self.assertEqual(policy["license_policy"], "production_eligible_training_data_offline_deployment_paused")
        self.assertTrue(policy["deployment_paused_by_user"])
        self.assertFalse(policy["iterative_test_access"])
        self.assertFalse(policy["frozen_video_used"])
        self.assertEqual(matrix["common"]["body_hierarchy"], "none")
        self.assertGreater(matrix["common"]["coarse_car_loss_weight"], 0)
        self.assertGreater(matrix["common"]["coarse_truck_loss_weight"], 0)

    def test_matrix_includes_full_production_checkpoint_continuation_at_224_and_256(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        candidates = {
            item["candidate_id"]: item for item in matrix["candidates"]
            if "CONTINUAL-DISTILL" in item["candidate_id"]
        }
        self.assertEqual({item["input_size"] for item in candidates.values()}, {224, 256})
        self.assertTrue(candidates)
        for item in candidates.values():
            self.assertEqual(item["architecture"], "mobilenet_v3_large")
            self.assertEqual(item["init_checkpoint"], item["teacher_checkpoint"])
            self.assertEqual(item["init_checkpoint_sha256"], item["teacher_checkpoint_sha256"])
            self.assertGreater(item["distill_weight"], 0)

    def test_dependency_hash_metadata_is_not_forwarded_to_trainer(self) -> None:
        args = MODULE.command_arguments({
            "candidate_id": "x", "purpose": "p", "init_checkpoint": "/x.pt",
            "init_checkpoint_sha256": "abc", "teacher_checkpoint_sha256": "def", "skip_test": True,
        })
        self.assertIn("--init-checkpoint", args)
        self.assertIn("--skip-test", args)
        self.assertNotIn("--init-checkpoint-sha256", args)
        self.assertNotIn("--teacher-checkpoint-sha256", args)

    def test_expected_quota_calculation_reflects_scene_weights(self) -> None:
        rows = [
            {"night": "true", "vehicle_size": "small", "occluded": "true", "stage66_licensed_adverse": "true", "color_supervised": "false"},
            {"night": "false", "vehicle_size": "large", "occluded": "false", "stage66_licensed_adverse": "false", "color_supervised": "false"},
        ]
        quotas = MODULE.expected_sampler_quotas(rows, {
            "night_sample_weight": 20, "small_sample_weight": 1, "occlusion_sample_weight": 1.5,
        })
        self.assertGreater(quotas["night"], 0.9)
        self.assertEqual(quotas["night"], quotas["licensed_adverse"])

    def test_evidence_chain_requires_final_manifest_and_fail_closed_policy(self) -> None:
        final_hash = "b" * 64
        base_hash = "a" * 64
        safe_policy = {
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        }
        manifest_report = {
            "status": "pass", "output_manifest_sha256": base_hash,
            "evaluation_membership_preserved": True, "cross_split_group_overlap": 0,
        }
        dedup_report = {
            "status": "pass", "input_manifest_sha256": base_hash,
            "output_manifest_sha256": "d" * 64, "unreadable_images": 0,
            "policy": {
                **safe_policy,
                "exact_byte_duplicates_removed_from_train": True,
                "exact_cross_split_train_duplicates_removed": True,
                "heldout_membership_preserved": True,
            },
        }
        near_dedup_report = {
            "status": "pass", "input_manifest_sha256": "d" * 64,
            "output_manifest_sha256": final_hash,
            "maximum_hamming_distance": 4,
            "residual_cross_split_near_pairs": 0,
            "residual_exact_sha_cross_split_overlap": 0,
            "policy": {
                **safe_policy,
                "all_train_rows_with_dhash_distance_le_4_to_heldout_removed": True,
                "heldout_membership_preserved": True,
                "heldout_labels_not_used_for_filtering": True,
            },
        }
        strata_report = {
            "status": "pass", "manifest_sha256": final_hash,
            "invalid_train_license_rows": 0, "frozen_marker_rows": 0,
            "policy": safe_policy,
        }
        leakage_report = {
            "status": "pass", "manifest_sha256": final_hash,
            "exact_sha_cross_split_overlap": 0, "source_group_cross_split_overlap": 0,
            "dhash": {"cross_split_near_pairs": 0}, "policy": safe_policy,
        }
        MODULE.validate_evidence_chain(
            final_hash, manifest_report, dedup_report, near_dedup_report, strata_report, leakage_report,
        )
        near_dedup_report["output_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "near-dedup"):
            MODULE.validate_evidence_chain(
                final_hash, manifest_report, dedup_report, near_dedup_report, strata_report, leakage_report,
            )


if __name__ == "__main__":
    unittest.main()
