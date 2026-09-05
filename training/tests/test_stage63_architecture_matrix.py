from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
MATRIX = TRAINING / "artifacts" / "attribute-domain-v2" / "stage63-architecture-domain-matrix.json"
LAUNCHER = TRAINING / "scripts" / "run_stage62_domain_consistency_matrix.py"
SPEC = importlib.util.spec_from_file_location("domain_launcher", LAUNCHER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage63ArchitectureMatrixTest(unittest.TestCase):
    def test_matrix_preserves_isolation_and_required_architectures(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        self.assertEqual(matrix["status"], "prepared_validation_only")
        self.assertTrue(matrix["common"]["skip_test"])
        self.assertTrue(matrix["execution_policy"]["deployment_paused_by_user"])
        self.assertFalse(matrix["execution_policy"]["frozen_video_used"])
        architectures = {candidate["architecture"] for candidate in matrix["candidates"]}
        self.assertIn("mobilenet_v3_large_dual", architectures)
        self.assertIn("mobilenet_v3_large_foreground_dual", architectures)
        self.assertIn("convnext_tiny", architectures)
        feature = next(
            candidate for candidate in matrix["candidates"]
            if candidate.get("unlabeled_consistency_mode") == "feature"
        )
        self.assertEqual(feature["architecture"], "mobilenet_v3_large")
        self.assertGreater(feature["unlabeled_consistency_weight"], 0)

    def test_distillation_dependency_is_metadata_not_a_training_flag(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        distill = next(candidate for candidate in matrix["candidates"] if "dependency" in candidate)
        self.assertEqual(distill["architecture"], "mobilenet_v3_large")
        self.assertGreater(distill["distill_weight"], 0)
        self.assertIn("dependency", MODULE.META_KEYS)
        arguments = MODULE.command_arguments(distill)
        self.assertNotIn("--dependency", arguments)
        self.assertIn("--distill-weight", arguments)


if __name__ == "__main__":
    unittest.main()
