import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decide_stage116_threshold_integrated_candidate.py"
SPEC = importlib.util.spec_from_file_location("stage116", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def color_variant(passing=True):
    return {
        "variant": "gate-best", "thresholds": {"white": 0.5}, "report_sha256": "c" * 64,
        "static": {"precision": 0.94 if passing else 0.92, "coverage": 0.5},
        "unknown_reduction": 0.3, "track_final": {"precision": 0.95, "coverage": 0.5},
        "stability": {"transition_stability": 0.99},
    }


def body_variant(passing=True):
    return {
        "variant": "best", "specialist_checkpoint": "/candidate.pt",
        "specialist_checkpoint_sha256": "b" * 64, "thresholds": {"bus": 0.8},
        "report_sha256": "d" * 64,
        "static": {"precision": 0.94, "coverage": 0.5 if passing else 0.44},
        "complex_coverage_gain": 0.2, "track_final": {"precision": 0.95, "coverage": 0.5},
        "stability": {"transition_stability": 0.99},
    }


def state(variant):
    return {"variants": [variant], "test_accessed": False, "frozen_video_used": False, "production_model_modified": False}


class Stage116DecisionTests(unittest.TestCase):
    def test_authorizes_only_when_both_components_pass(self):
        result = MODULE.decide(state(color_variant()), state(body_variant()))
        self.assertTrue(result["independent_test_authorized"])
        self.assertEqual(result["status"], "pass_independent_test_authorized")

    def test_body_failure_closes_authorization(self):
        result = MODULE.decide(state(color_variant()), state(body_variant(False)))
        self.assertFalse(result["independent_test_authorized"])
        self.assertIsNone(result["candidate"])

    def test_color_failure_closes_authorization(self):
        result = MODULE.decide(state(color_variant(False)), state(body_variant()))
        self.assertFalse(result["independent_test_authorized"])

    def test_test_access_flag_raises(self):
        bad = state(color_variant()); bad["test_accessed"] = True
        with self.assertRaises(RuntimeError):
            MODULE.decide(bad, state(body_variant()))


if __name__ == "__main__":
    unittest.main()
