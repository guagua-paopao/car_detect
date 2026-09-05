from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage84_v2_body_after_gpu_chain.py"
SPEC = importlib.util.spec_from_file_location("stage84_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage84RunnerTests(unittest.TestCase):
    def test_dependency_state_machine(self) -> None:
        self.assertEqual(MODULE.stage76_action("running"), "wait")
        self.assertEqual(MODULE.stage76_action("complete_validation_only_two_phase"), "ready")
        self.assertEqual(MODULE.stage76_action("failed_closed"), "fail")
        self.assertEqual(MODULE.stage80_action("waiting_for_stage77"), "wait")
        self.assertEqual(MODULE.stage80_action("complete_validation_only"), "ready")
        self.assertEqual(MODULE.stage80_action("failed_closed_runtime"), "ready")

    def test_select_stage76_checkpoint_binds_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            state = {
                "status": "complete_validation_only_two_phase",
                "completed": [{
                    "phase": "cctv_finetune", "best_checkpoint": str(checkpoint),
                    "best_checkpoint_sha256": MODULE.sha256(checkpoint),
                }],
            }
            path, digest = MODULE.select_stage76_checkpoint(state)
            self.assertEqual(path, checkpoint)
            self.assertEqual(digest, MODULE.sha256(checkpoint))

    def test_command_arguments_keeps_skip_test(self) -> None:
        arguments = MODULE.command_arguments({
            "candidate_id": "ignored", "skip_test": True, "epochs": 8,
            "color_loss_weight": 0.0,
        })
        self.assertIn("--skip-test", arguments)
        self.assertEqual(arguments[arguments.index("--epochs") + 1], "8")


if __name__ == "__main__":
    unittest.main()
