from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "recover_stage71_validation_chain_v3.py"
SPEC = importlib.util.spec_from_file_location("stage71_chain_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage71RecoveredChainV3Test(unittest.TestCase):
    def test_only_final_launcher_is_replaced(self) -> None:
        original = (
            ("student_validation", Path("student"), "a"),
            ("fail_closed_track", Path("v2"), "b"),
            ("final_test", Path("old-test"), "c"),
            ("onnx_backend", Path("onnx"), "d"),
        )
        updated = MODULE.replace_final_launcher(original)
        names = [item[0] for item in updated]
        self.assertEqual(names, [item[0] for item in original])
        final = next(item for item in updated if item[0] == "final_test")
        self.assertEqual(final[1].name, "run_stage71_final_test_after_fail_closed_v3.sh")
        self.assertEqual(updated[-1], original[-1])

    def test_missing_final_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no final-test"):
            MODULE.replace_final_launcher((("onnx_backend", Path("onnx"), "d"),))

    def test_launcher_pins_v3_wrapper(self) -> None:
        launcher = (
            TRAINING_ROOT / "scripts" / "launch_stage71_recovered_chain_v3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("recover_stage71_validation_chain_v3.py", launcher)
        self.assertIn("3721ab2c3e64cb2059e063bac14cc169658deb7cbd4ef1246386fe2428a4d97d", launcher)
        self.assertNotIn("vcas_rtsp_demo_60s", launcher)


if __name__ == "__main__":
    unittest.main()
