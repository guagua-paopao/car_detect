from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage76_body_pretrain_finetune.py"
SPEC = importlib.util.spec_from_file_location("stage76_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage76RunnerTests(unittest.TestCase):
    def test_command_arguments_keep_skip_test_and_drop_metadata(self) -> None:
        args = MODULE.command_arguments({
            "candidate_id": "candidate",
            "phase": "dvm_pretrain",
            "epochs": 5,
            "skip_test": True,
            "pretrained": False,
        })
        self.assertIn("--skip-test", args)
        self.assertIn("--epochs", args)
        self.assertNotIn("--candidate-id", args)
        self.assertNotIn("--phase", args)
        self.assertNotIn("--pretrained", args)

    def test_truthy_is_explicit(self) -> None:
        self.assertTrue(MODULE.truthy("true"))
        self.assertTrue(MODULE.truthy("1"))
        self.assertFalse(MODULE.truthy("unknown"))
        self.assertFalse(MODULE.truthy("false"))


if __name__ == "__main__":
    unittest.main()
