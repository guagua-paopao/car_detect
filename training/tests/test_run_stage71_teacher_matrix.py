from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage71_teacher_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_stage71_teacher_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage71TeacherRunnerTests(unittest.TestCase):
    def test_command_arguments_exclude_evidence_metadata(self) -> None:
        arguments = MODULE.command_arguments({
            "candidate_id": "teacher",
            "specialist": "color",
            "init_checkpoint_sha256": "abc",
            "skip_test": True,
            "learning_rate": 1e-4,
        })
        self.assertEqual(arguments, ["--skip-test", "--learning-rate", "0.0001"])

    def test_expected_quota_applies_trainer_weight_clamp_and_night_multiplier(self) -> None:
        rows = [
            {"sample_weight": "1", "night": "false", "stage71_origin": "dvm", "color_supervised": "true", "color": "white"},
            {"sample_weight": "10", "night": "true", "stage71_origin": "cctv", "color_supervised": "true", "color": "red"},
        ]
        quota = MODULE.expected_supervised_quota(rows, {"night_sample_weight": 100})
        self.assertAlmostEqual(quota["night"], 1000 / 1001)
        self.assertAlmostEqual(quota["cctv"], 1000 / 1001)

    def test_unlabeled_stream_rejects_known_target(self) -> None:
        rows = [{"split": "train", "body_type": "sedan", "color": "unknown", "body_type_supervised": "false", "color_supervised": "false", "night": "true"}]
        with self.assertRaisesRegex(RuntimeError, "contains a class target"):
            MODULE.expected_unlabeled_night(rows, 20)


if __name__ == "__main__":
    unittest.main()
