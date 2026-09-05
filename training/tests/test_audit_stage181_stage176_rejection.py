from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage181_stage176_rejection.py"
SPEC = importlib.util.spec_from_file_location("stage181", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage181Tests(unittest.TestCase):
    def test_truthy_is_strict(self) -> None:
        self.assertTrue(MODULE.truthy("true"))
        self.assertFalse(MODULE.truthy("unknown"))
        self.assertFalse(MODULE.truthy("false"))


if __name__ == "__main__":
    unittest.main()
