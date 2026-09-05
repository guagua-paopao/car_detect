import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage196", ROOT / "scripts" / "prepare_stage196_abtd_body_teacher_input.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["stage196"] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage196PrepareTest(unittest.TestCase):
    def test_truthy_is_conservative(self):
        self.assertTrue(MODULE.truthy("true"))
        self.assertTrue(MODULE.truthy("1"))
        self.assertFalse(MODULE.truthy("unknown"))
        self.assertFalse(MODULE.truthy(""))


if __name__ == "__main__":
    unittest.main()
