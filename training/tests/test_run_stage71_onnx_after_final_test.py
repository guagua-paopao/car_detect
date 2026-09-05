from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stage71_onnx_after_final_test.py"
)
SPEC = importlib.util.spec_from_file_location("stage71_onnx_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage71OnnxRunnerTests(unittest.TestCase):
    def test_accepts_passing_final_state_without_polling_tmux(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(
                json.dumps({"status": "pass_backend_eligible"}),
                encoding="utf-8",
            )
            result = MODULE.wait_for_final_test(state, "does-not-exist", 0.01)
            self.assertEqual(result["status"], "pass_backend_eligible")

    def test_rejects_failed_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(
                json.dumps({"status": "fail_closed_before_backend"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                MODULE.wait_for_final_test(state, "does-not-exist", 0.01)


if __name__ == "__main__":
    unittest.main()
