import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_stage215_body_pseudolabel_rule.py"
SPEC = importlib.util.spec_from_file_location("stage215_body_rule", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
validate_checkpoint_body_ontology = MODULE.validate_checkpoint_body_ontology


V1 = [
    "sedan", "suv", "mpv", "van", "pickup", "bus", "light_truck",
    "heavy_truck", "other", "unknown",
]
V2 = [
    "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
    "light_truck", "heavy_truck", "other", "unknown",
]


class CheckpointBodyOntologyTests(unittest.TestCase):
    def test_legacy_v1_subset_is_allowed_by_exact_label_name(self):
        self.assertEqual(validate_checkpoint_body_ontology(V1, V2, "legacy"), V1)

    def test_v2_contract_is_allowed(self):
        self.assertEqual(validate_checkpoint_body_ontology(V2, V2, "current"), V2)

    def test_unknown_label_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "no unknown"):
            validate_checkpoint_body_ontology(V1[:-1], V2, "bad")

    def test_unsupported_label_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            validate_checkpoint_body_ontology(V1 + ["coupe"], V2, "bad")

    def test_duplicate_label_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            validate_checkpoint_body_ontology(V1 + ["sedan"], V2, "bad")


if __name__ == "__main__":
    unittest.main()
