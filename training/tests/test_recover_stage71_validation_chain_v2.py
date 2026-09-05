from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "recover_stage71_validation_chain_v2.py"
SPEC = importlib.util.spec_from_file_location("stage71_chain_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage71RecoveredChainV2Test(unittest.TestCase):
    def test_fail_closed_stage_precedes_test(self) -> None:
        class Fake:
            LAUNCHERS = (
                ("teacher_validation", Path("teacher"), "a"),
                ("student_training", Path("students"), "b"),
                ("student_validation", Path("validation"), "c"),
                ("final_test", Path("test"), "d"),
                ("onnx_backend", Path("onnx"), "e"),
            )

        launchers = MODULE.patched_launchers(Fake)
        names = [item[0] for item in launchers]
        self.assertEqual(
            names,
            [
                "teacher_validation",
                "student_training",
                "student_validation",
                "fail_closed_track",
                "final_test",
                "onnx_backend",
            ],
        )

    def test_final_test_launcher_is_v2_guarded(self) -> None:
        class Fake:
            LAUNCHERS = (
                ("teacher_validation", Path("teacher"), "a"),
                ("student_training", Path("students"), "b"),
                ("student_validation", Path("validation"), "c"),
                ("final_test", Path("test"), "d"),
                ("onnx_backend", Path("onnx"), "e"),
            )

        launchers = {item[0]: item for item in MODULE.patched_launchers(Fake)}
        self.assertEqual(
            launchers["final_test"][1].name,
            "run_stage71_final_test_after_fail_closed_v2.sh",
        )
        self.assertIn("fail_closed_pair_gate", launchers["fail_closed_track"][1].name)

    def test_launcher_pins_wrapper_and_does_not_touch_frozen_video(self) -> None:
        launcher = (
            TRAINING_ROOT / "scripts" / "launch_stage71_recovered_chain_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("recover_stage71_validation_chain_v2.py", launcher)
        self.assertIn("0ff1f89982f08a33cfafff66495100b63282d57c7bdc4db430ce2196bdb0aa22", launcher)
        self.assertNotIn("vcas_rtsp_demo_60s", launcher)


if __name__ == "__main__":
    unittest.main()
