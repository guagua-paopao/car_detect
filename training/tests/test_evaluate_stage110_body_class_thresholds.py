import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage110", SCRIPTS / "evaluate_stage110_body_class_thresholds.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage110BodyThresholdTests(unittest.TestCase):
    def test_optimizer_can_abstain_unreliable_light_truck(self):
        predictions = ["bus"] * 10 + ["light_truck"] * 4
        confidences = [0.60] * 10 + [0.99, 0.90, 0.80, 0.70]
        truths = ["bus"] * 10 + ["heavy_truck"] * 4
        result = MODULE.thresholding.optimize_thresholds(
            predictions, confidences, truths, 0.93
        )
        self.assertEqual(result["selected"], 10)
        self.assertEqual(result["thresholds"]["light_truck"], 1.01)

    def test_metric_never_relabels(self):
        result = MODULE.thresholding.metric(
            ["bus", "heavy_truck"],
            [0.7, 0.8],
            ["bus", "light_truck"],
            {"bus": 0.5, "heavy_truck": 0.9},
        )
        self.assertEqual(result["selected"], 1)
        self.assertEqual(result["correct"], 1)


if __name__ == "__main__":
    unittest.main()
