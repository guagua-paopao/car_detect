from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_stage71_body_teacher.py"
SPEC = importlib.util.spec_from_file_location("recover_stage71_body_teacher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def state(*, completed_specialist: str = "color", active_specialist: str = "body") -> dict:
    return {
        "status": "running",
        "completed": [{"specialist": completed_specialist}],
        "active": {"specialist": active_specialist},
        "failed": [],
    }


class Stage71BodyRecoveryTest(unittest.TestCase):
    def test_accepts_color_complete_body_interrupted_state(self) -> None:
        MODULE.validate_interrupted_body_state(state())

    def test_rejects_wrong_completed_or_active_specialist(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not color"):
            MODULE.validate_interrupted_body_state(state(completed_specialist="body"))
        with self.assertRaisesRegex(RuntimeError, "not body"):
            MODULE.validate_interrupted_body_state(state(active_specialist="color"))

    def test_rejects_failed_or_multiple_completed_state(self) -> None:
        failed = state()
        failed["failed"] = [{"error": "x"}]
        with self.assertRaisesRegex(RuntimeError, "clean running"):
            MODULE.validate_interrupted_body_state(failed)
        multiple = state()
        multiple["completed"].append({"specialist": "body"})
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            MODULE.validate_interrupted_body_state(multiple)

    def test_completed_record_must_remain_sha_bound(self) -> None:
        record = {
            "candidate_id": "color", "specialist": "color", "last_epoch": 9,
            "best_checkpoint_sha256": "a", "metrics_sha256": "b",
            "test_metrics_sha256": "c", "model_card_sha256": "d",
            "training_log_sha256": "e",
            "last_checkpoint": {"sha256": "f", "epoch": 9, "best_score": 0.9},
        }
        MODULE.require_completed_record_matches(record, dict(record))
        changed = {**record, "metrics_sha256": "changed"}
        with self.assertRaisesRegex(RuntimeError, "metrics_sha256"):
            MODULE.require_completed_record_matches(record, changed)


if __name__ == "__main__":
    unittest.main()
