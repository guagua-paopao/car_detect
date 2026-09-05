import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage193", ROOT / "scripts" / "audit_stage193_abtd_metadata_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage193Test(unittest.TestCase):
    def test_exact_mapping_is_conservative(self):
        self.assertEqual(MODULE.EXACT_BODY, {1: "bus", 2: "truck", 5: "van"})
        self.assertNotIn(0, MODULE.EXACT_BODY)
        self.assertNotIn(9, MODULE.EXACT_BODY)

    def test_pinned_integrity_contract(self):
        self.assertEqual(MODULE.EXPECTED["labels.zip"][0], 2233642)
        self.assertEqual(len(MODULE.EXPECTED["metadata.csv"][1]), 64)


if __name__ == "__main__":
    unittest.main()
