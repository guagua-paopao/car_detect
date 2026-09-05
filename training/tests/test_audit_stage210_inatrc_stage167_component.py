import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage210_inatrc_stage167_component.py"
SPEC = importlib.util.spec_from_file_location("stage210", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage210Tests(unittest.TestCase):
    def test_hamming_index_finds_radius_four_only(self):
        index = MODULE.HammingIndex()
        index.add(0, "base", "train", "g0")
        self.assertIsNotNone(index.nearest(0b1111, 4))
        self.assertIsNone(index.nearest(0b11111, 4))

    def test_stage91_policy_is_fail_closed(self):
        valid = {
            "split": "train", "source_dataset": "InaTRC-v1", "source_license": "CC-BY-4.0",
            "license_train_eligible": "true", "body_type": "truck", "body_type_supervised": "true",
            "color": "unknown", "color_supervised": "false",
        }
        self.assertEqual(MODULE.stage91_row_valid(valid), (True, ""))
        invalid = dict(valid, source_license="unknown")
        self.assertEqual(MODULE.stage91_row_valid(invalid)[0], False)
        invalid = dict(valid, color="red", color_supervised="true")
        self.assertEqual(MODULE.stage91_row_valid(invalid)[0], False)


if __name__ == "__main__":
    unittest.main()
