from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = Path(os.environ.get(
    "STUDENT_MATERIALIZER_SCRIPT",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_stage71_student_matrix.py",
))
SPEC = importlib.util.spec_from_file_location("prepare_stage71_students", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def report(pairs, passing):
    return {
        "status": "pass_pairs_available",
        "passing_pairs": passing,
        "pairs": pairs,
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }


class PrepareStage71StudentMatrixTests(unittest.TestCase):
    def test_selects_best_all_green_pair(self) -> None:
        low = {
            "pair_id": "low", "screen_status": "pass", "gates": {"a": True},
            "hard": {"body_type": {"coverage": 0.5}, "color": {"coverage": 0.3}},
        }
        high = {
            "pair_id": "high", "screen_status": "pass", "gates": {"a": True},
            "hard": {"body_type": {"coverage": 0.7}, "color": {"coverage": 0.4}},
        }
        self.assertEqual(MODULE.select_passing_pair(report([low, high], ["low", "high"]))["pair_id"], "high")

    def test_rejects_report_without_passing_pair(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no passing pair"):
            MODULE.select_passing_pair({"status": "complete_all_pairs_rejected", "passing_pairs": []})

    def test_rejects_passing_id_with_failed_gate(self) -> None:
        pair = {"pair_id": "bad", "screen_status": "pass", "gates": {"a": True, "b": False}}
        with self.assertRaisesRegex(RuntimeError, "all-green"):
            MODULE.select_passing_pair(report([pair], ["bad"]))

    def test_binds_only_matching_specialist_teacher(self) -> None:
        pair = {
            "body_checkpoint": "/body.pt", "body_checkpoint_sha256": "a" * 64,
            "color_checkpoint": "/color.pt", "color_checkpoint_sha256": "b" * 64,
        }
        body = {"specialist": "body"}
        color = {"specialist": "color"}
        MODULE.bind_teacher(body, pair)
        MODULE.bind_teacher(color, pair)
        self.assertEqual(body["body_teacher_checkpoint"], "/body.pt")
        self.assertNotIn("color_teacher_checkpoint", body)
        self.assertEqual(color["color_teacher_checkpoint"], "/color.pt")
        self.assertNotIn("body_teacher_checkpoint", color)


if __name__ == "__main__":
    unittest.main()
