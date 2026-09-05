from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_stage71_teacher_matrix.py"
SPEC = importlib.util.spec_from_file_location("recover_stage71_teacher_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage71RecoveryTests(unittest.TestCase):
    def test_replaces_exactly_one_init_checkpoint_and_preserves_test_lock(self) -> None:
        command = [
            "/python",
            "/train.py",
            "--skip-test",
            "--init-checkpoint",
            "/init.pt",
            "--epochs",
            "10",
        ]
        recovered = MODULE.replace_init_with_resume(command, Path("/run/last.pt"))
        self.assertNotIn("--init-checkpoint", recovered)
        self.assertIn("--skip-test", recovered)
        self.assertEqual(
            recovered[-2:],
            ["--resume", str(Path("/run/last.pt").resolve())],
        )

    def test_rejects_command_without_test_lock(self) -> None:
        command = ["/python", "/train.py", "--init-checkpoint", "/init.pt"]
        with self.assertRaisesRegex(RuntimeError, "lost --skip-test"):
            MODULE.replace_init_with_resume(command, Path("/run/last.pt"))

    def test_rejects_state_that_accessed_test(self) -> None:
        state = {
            "test_accessed": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        }
        with self.assertRaisesRegex(RuntimeError, "test_accessed"):
            MODULE.validate_locked_flags(state)

    def test_accepts_locked_state(self) -> None:
        state = {
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        }
        MODULE.validate_locked_flags(state)


if __name__ == "__main__":
    unittest.main()
