import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


for name in (
    "evaluate_stage109_color_class_thresholds",
    "evaluate_stage159_component_validation",
    "evaluate_stage166_selective_short_tracks",
    "evaluate_v2_decoupled_shared_validation",
):
    sys.modules.setdefault(name, types.ModuleType(name))

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_stage171_fresh_holdout_once.py"
SPEC = importlib.util.spec_from_file_location("stage171_once", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage171Tests(unittest.TestCase):
    def test_threshold_validation_is_fail_closed(self):
        self.assertEqual(MODULE.valid_thresholds({"suv": 0.8}, "body"), {"suv": 0.8})
        for value in ({}, {"suv": 0.0}, {"suv": 1.02}):
            with self.assertRaises(RuntimeError):
                MODULE.valid_thresholds(value, "body")

    def test_track_order_prefers_numeric_frame_index(self):
        rows = [
            {"stage170_group": "g", "frame_index": "10", "body_type_supervised": "true"},
            {"stage170_group": "g", "frame_index": "2", "body_type_supervised": "true"},
        ]
        self.assertEqual(MODULE.grouped_indexes(rows, "body"), {"g": [1, 0]})

    def test_track_truth_conflict_fails_closed(self):
        rows = [{"body_type": "suv"}, {"body_type": "sedan"}]
        with self.assertRaisesRegex(RuntimeError, "inconsistent body_type"):
            MODULE.single_track_truth(rows, [0, 1], "body_type")

    def test_post_test_authorization_never_allows_frozen_or_deploy(self):
        self.assertEqual(
            MODULE.post_test_authorization(True),
            {"onnx_backend_gates": True, "frozen_video_replay": False, "deployment": False},
        )
        self.assertFalse(MODULE.post_test_authorization(False)["onnx_backend_gates"])

    def test_load_stage170_rejects_consumed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("split\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps({
                "status": "pass_fresh_holdout_built_and_sealed",
                "output_manifest_sha256": MODULE.sha256(manifest),
            }), encoding="utf-8")
            state = root / "state.json"
            state.write_text(json.dumps({
                "status": "fresh_holdout_sealed_pending_one_time_inference",
                "authorization": {
                    "one_time_independent_test_inference": True,
                    "onnx_backend_gates": False,
                    "frozen_video_replay": False,
                    "deployment": False,
                },
                "test_inference_run": True,
                "test_used_for_selection": False,
                "threshold_or_temperature_search": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
                "manifest": str(manifest),
                "manifest_sha256": MODULE.sha256(manifest),
                "report": str(report),
                "report_sha256": MODULE.sha256(report),
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.load_stage170(state, MODULE.sha256(state))

    def test_claim_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                output.mkdir(parents=True, exist_ok=False)


if __name__ == "__main__":
    unittest.main()
