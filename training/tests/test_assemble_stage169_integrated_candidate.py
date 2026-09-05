import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "assemble_stage169_integrated_candidate.py"
SPEC = importlib.util.spec_from_file_location("stage169", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IntegratedCandidateTests(unittest.TestCase):
    def labels(self):
        return {"body_types": ["sedan", "unknown"], "colors": ["black", "unknown"]}

    def test_matching_stretch_geometry_passes(self):
        result = MODULE.validate_geometry_contract(
            {"input_size": 256, "resize_mode": "stretch", "body_types": ["sedan", "unknown"]},
            {"input_size": 256, "resize_mode": "stretch", "color_types": ["black", "unknown"]},
            self.labels(),
        )
        self.assertTrue(result["current_adapter_compatible"])

    def test_letterbox_body_fails_current_adapter(self):
        with self.assertRaisesRegex(RuntimeError, "stretch/stretch"):
            MODULE.validate_geometry_contract(
                {"input_size": 256, "resize_mode": "letterbox", "body_types": ["sedan", "unknown"]},
                {"input_size": 256, "resize_mode": "stretch", "color_types": ["black", "unknown"]},
                self.labels(),
            )

    def test_taxonomy_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "body checkpoint taxonomy"):
            MODULE.validate_geometry_contract(
                {"input_size": 256, "resize_mode": "stretch", "body_types": ["other", "unknown"]},
                {"input_size": 256, "resize_mode": "stretch", "color_types": ["black", "unknown"]},
                self.labels(),
            )

    def test_pass_authorizes_only_fresh_holdout_build(self):
        authorization = MODULE.next_stage_authorization(True)
        self.assertTrue(authorization["stage160_new_holdout_build"])
        self.assertFalse(authorization["independent_test_inference"])
        self.assertFalse(authorization["onnx_backend_gates"])
        self.assertFalse(authorization["frozen_video_replay"])
        self.assertFalse(authorization["deployment"])

    def test_rejection_authorizes_nothing(self):
        self.assertFalse(any(MODULE.next_stage_authorization(False).values()))


if __name__ == "__main__":
    unittest.main()
