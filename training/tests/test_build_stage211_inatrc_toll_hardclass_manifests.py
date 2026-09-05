import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage211_inatrc_toll_hardclass_manifests.py"
SPEC = importlib.util.spec_from_file_location("stage211", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage211Tests(unittest.TestCase):
    def test_complex_weak_passenger_multiplier(self):
        row = {"split": "train", "body_type_supervised": "true", "body_type": "suv", "small_target": "true"}
        self.assertEqual(MODULE.passenger_multiplier(row), 1.25)
        row["small_target"] = "false"
        self.assertEqual(MODULE.passenger_multiplier(row), 1.10)

    def test_validation_and_unsupervised_never_reweighted(self):
        validation = {"split": "validation", "body_type_supervised": "true", "body_type": "mpv", "night": "true"}
        unlabeled = {"split": "train", "body_type_supervised": "false", "body_type": "suv", "night": "true"}
        self.assertEqual(MODULE.passenger_multiplier(validation), 1.0)
        self.assertEqual(MODULE.passenger_multiplier(unlabeled), 1.0)

    def test_weight_is_bounded(self):
        self.assertEqual(MODULE.weighted("9", 2), "10.000000")
        self.assertEqual(MODULE.weighted("0.01", 0.5), "0.100000")


if __name__ == "__main__":
    unittest.main()
