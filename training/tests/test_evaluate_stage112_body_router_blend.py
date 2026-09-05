import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage112", SCRIPTS / "evaluate_stage112_body_router_blend.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage112BlendTests(unittest.TestCase):
    def test_non_family_prediction_is_unchanged(self):
        result = MODULE.blended_output(
            {"choose_family": False, "main_label": "bus", "main_confidence": 0.8},
            0.5,
        )
        self.assertEqual(result, {"body_label": "bus", "body_confidence": 0.8})

    def test_alpha_endpoints_use_expected_subtype_evidence(self):
        record = {
            "choose_family": True,
            "family_mass": 0.81,
            "main_light_conditional": 0.8,
            "main_heavy_conditional": 0.2,
            "specialist_light": 0.1,
            "specialist_heavy": 0.85,
        }
        self.assertEqual(MODULE.blended_output(record, 0.0)["body_label"], "light_truck")
        self.assertEqual(MODULE.blended_output(record, 1.0)["body_label"], "heavy_truck")

    def test_invalid_alpha_fails_closed(self):
        with self.assertRaises(ValueError):
            MODULE.blended_output({}, 1.1)


if __name__ == "__main__":
    unittest.main()
