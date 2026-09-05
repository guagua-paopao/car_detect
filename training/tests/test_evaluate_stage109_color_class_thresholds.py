import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage109", SCRIPTS / "evaluate_stage109_color_class_thresholds.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage109ThresholdTests(unittest.TestCase):
    def test_metric_uses_predicted_class_threshold(self):
        result = MODULE.metric(
            ["white", "black", "white"],
            [0.60, 0.90, 0.80],
            ["white", "white", "white"],
            {"white": 0.50, "black": 0.95},
        )
        self.assertEqual(result["selected"], 2)
        self.assertEqual(result["correct"], 2)

    def test_optimizer_releases_safe_class_and_abstains_unsafe_class(self):
        predictions = ["white"] * 8 + ["black"] * 4
        confidences = [0.60] * 8 + [0.99, 0.90, 0.80, 0.70]
        truths = ["white"] * 8 + ["white"] * 4
        result = MODULE.optimize_thresholds(predictions, confidences, truths, 0.93)
        self.assertEqual(result["selected"], 8)
        self.assertEqual(result["thresholds"]["black"], 1.01)
        self.assertLessEqual(result["thresholds"]["white"], 0.60)

    def test_optimizer_finds_global_precision_tradeoff(self):
        predictions = ["green"] * 10 + ["red"] * 2
        confidences = [0.70] * 10 + [0.95, 0.90]
        truths = ["green"] * 10 + ["red", "green"]
        result = MODULE.optimize_thresholds(predictions, confidences, truths, 0.90)
        self.assertEqual(result["selected"], 12)
        self.assertGreaterEqual(result["precision"], 0.90)

    def test_invalid_precision_target_fails_closed(self):
        with self.assertRaises(ValueError):
            MODULE.optimize_thresholds(["white"], [0.9], ["white"], 0.0)


if __name__ == "__main__":
    unittest.main()
