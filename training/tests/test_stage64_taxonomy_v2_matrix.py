from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
LOCAL_MATRIX = TRAINING / "artifacts" / "attribute-domain-v2" / "stage64-taxonomy-v2-matrix.json"
REMOTE_MATRIX = TRAINING.parents[1] / "runs" / "attributes" / "plans" / "stage64-taxonomy-v2-matrix.json"
MATRIX = LOCAL_MATRIX if LOCAL_MATRIX.is_file() else REMOTE_MATRIX
RUNNER = TRAINING / "scripts" / "run_stage64_taxonomy_v2_matrix.py"
SPEC = importlib.util.spec_from_file_location("stage64_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage64TaxonomyV2MatrixTest(unittest.TestCase):
    def test_matrix_is_offline_validation_only_and_hierarchical(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        policy = matrix["execution_policy"]
        self.assertEqual(matrix["status"], "prepared_validation_only")
        self.assertEqual(matrix["runner_script_sha256"], MODULE.sha256(RUNNER).upper())
        for key in (
            "manifest_sha256", "manifest_report_sha256",
            "unlabeled_manifest_sha256", "unlabeled_report_sha256",
        ):
            self.assertRegex(matrix[key], r"^[A-F0-9]{64}$")
        self.assertEqual(len(matrix["source_hashes"]), 4)
        for digest in matrix["source_hashes"].values():
            self.assertRegex(digest, r"^[A-F0-9]{64}$")
        self.assertFalse(policy["iterative_test_access"])
        self.assertFalse(policy["frozen_video_used"])
        self.assertTrue(policy["deployment_paused_by_user"])
        self.assertEqual(policy["license_policy"], "offline_research_only_no_deployment")
        self.assertTrue(policy["exact_body_gate_not_replaced_by_family_metric"])
        self.assertEqual(matrix["common"]["body_hierarchy"], "truck_family")
        self.assertTrue(matrix["common"]["skip_test"])

    def test_required_candidate_variants_are_present(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        architectures = {candidate["architecture"] for candidate in matrix["candidates"]}
        self.assertIn("mobilenet_v3_large", architectures)
        self.assertIn("mobilenet_v3_large_foreground_dual", architectures)
        feature = next(
            candidate for candidate in matrix["candidates"]
            if candidate.get("unlabeled_consistency_weight", 0) > 0
        )
        self.assertEqual(feature["unlabeled_consistency_mode"], "feature")
        self.assertGreater(feature["color_sample_weight"], 1.0)

    def test_runner_never_forwards_metadata_as_training_flags(self) -> None:
        args = MODULE.command_arguments({
            "candidate_id": "x",
            "purpose": "test",
            "init_checkpoint_sha256": "a" * 64,
            "body_hierarchy": "truck_family",
            "skip_test": True,
        })
        self.assertNotIn("--candidate-id", args)
        self.assertNotIn("--purpose", args)
        self.assertNotIn("--init-checkpoint-sha256", args)
        self.assertIn("--body-hierarchy", args)
        self.assertIn("--skip-test", args)

    def test_unlabeled_evidence_validation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "attribute-proposals.csv"
            report = root / "combine-report.json"
            manifest.write_text("split,body_type,color\ntrain,unknown,unknown\n", encoding="utf-8")
            payload = {
                "status": "pass",
                "output_manifest_sha256": MODULE.sha256(manifest),
                "policy": {
                    "all_inputs_hash_verified": True,
                    "all_rows_train_only": True,
                    "all_attributes_remain_unknown_unsupervised": True,
                    "duplicate_image_paths_rejected": True,
                    "validation_or_test_used": False,
                    "frozen_video_used": False,
                    "production_model_modified": False,
                    "deployment_performed": False,
                },
            }
            report.write_text(json.dumps(payload), encoding="utf-8")
            matrix = {
                "unlabeled_manifest": str(manifest),
                "unlabeled_manifest_sha256": MODULE.sha256(manifest),
                "unlabeled_report": str(report),
                "unlabeled_report_sha256": MODULE.sha256(report),
            }
            self.assertEqual(MODULE.validate_unlabeled_evidence(matrix), (manifest, report))
            payload["policy"]["all_attributes_remain_unknown_unsupervised"] = False
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.validate_unlabeled_evidence(matrix)


if __name__ == "__main__":
    unittest.main()
