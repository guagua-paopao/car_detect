from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_stage71_validation_chain.py"
SPEC = importlib.util.spec_from_file_location("recover_stage71_validation_chain", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage71ValidationChainTests(unittest.TestCase):
    def test_within_accepts_child_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            child = root / "child"
            child.mkdir()
            self.assertEqual(MODULE.within(child, root), child.resolve())
            with self.assertRaises(ValueError):
                MODULE.within(root.parent, root)

    def test_gate_closed_keeps_stage_and_status(self) -> None:
        error = MODULE.GateClosed("student_validation", "complete_all_pairs_rejected")
        self.assertEqual(error.stage, "student_validation")
        self.assertEqual(error.status, "complete_all_pairs_rejected")


if __name__ == "__main__":
    unittest.main()
