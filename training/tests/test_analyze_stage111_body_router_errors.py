import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage111", SCRIPTS / "analyze_stage111_body_router_errors.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage111RouterAnalysisTests(unittest.TestCase):
    def test_diagnoses_specialist_when_family_route_is_good(self):
        records = []
        for index in range(10):
            records.append(
                {
                    "truth": "light_truck",
                    "choose_family": True,
                    "main_argmax": "truck",
                    "routed": "heavy_truck",
                    "specialist_argmax": "light_truck" if index < 4 else "heavy_truck",
                    "family_mass": 0.9,
                    "specialist_best_probability": 0.8,
                }
            )
        result = MODULE.summarize(records)
        self.assertEqual(result["diagnosis"], "light_heavy_specialist_domain_repair_required")

    def test_diagnoses_both_when_family_route_and_specialist_fail(self):
        records = []
        for index in range(10):
            records.append(
                {
                    "truth": "light_truck",
                    "choose_family": index < 5,
                    "main_argmax": "truck" if index < 5 else "bus",
                    "routed": "heavy_truck" if index < 5 else "bus",
                    "specialist_argmax": "heavy_truck",
                    "family_mass": 0.5,
                    "specialist_best_probability": 0.8,
                }
            )
        result = MODULE.summarize(records)
        self.assertEqual(
            result["diagnosis"],
            "main_family_router_and_specialist_domain_repair_required",
        )


if __name__ == "__main__":
    unittest.main()
