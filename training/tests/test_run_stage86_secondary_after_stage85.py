from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage86_secondary_after_stage85.py"
SPEC = importlib.util.spec_from_file_location("stage86_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage86RunnerTests(unittest.TestCase):
    def test_primary_action_is_fail_closed(self) -> None:
        self.assertEqual(MODULE.primary_action("waiting_for_stage80_stage84"), "wait")
        self.assertEqual(MODULE.primary_action("running_validation"), "wait")
        self.assertEqual(MODULE.primary_action("complete_validation_only_shared_pass_pending_evidence"), "ready")
        self.assertEqual(MODULE.primary_action("rejected_validation_gate_failure"), "reject")
        self.assertEqual(MODULE.primary_action("unexpected"), "fail")


if __name__ == "__main__":
    unittest.main()
