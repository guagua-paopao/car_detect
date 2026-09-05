import importlib.util
from pathlib import Path
import sys
import types
import unittest


stub = types.ModuleType("evaluate_stage159_component_validation")
sys.modules.setdefault("evaluate_stage159_component_validation", stub)
SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_stage165_track_frontier.py"
SPEC = importlib.util.spec_from_file_location("stage165_frontier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrackFrontierTests(unittest.TestCase):
    def config(self, **updates):
        value = {
            "window": 3,
            "minimum_observations": 1,
            "entry_share": 0.60,
            "entry_margin": 0.0,
            "switch_share": 0.70,
            "switch_margin": 0.15,
            "switch_patience": 2,
            "unknown_hold": 1,
        }
        value.update(updates)
        return value

    def test_high_confidence_singleton_is_admitted(self):
        self.assertEqual(
            MODULE.frontier_fuse([("suv", 0.96, 1.0)], **self.config()),
            ["suv"],
        )

    def test_unknown_hold_is_bounded(self):
        outputs = MODULE.frontier_fuse(
            [("suv", 0.96, 1.0), ("unknown", 0.0, 0.3), ("unknown", 0.0, 0.2)],
            **self.config(unknown_hold=1),
        )
        self.assertEqual(outputs, ["suv", "suv", "unknown"])

    def test_ambiguous_known_conflict_is_unknown(self):
        outputs = MODULE.frontier_fuse(
            [("suv", 0.96, 1.0), ("mpv", 0.96, 1.0)],
            **self.config(switch_share=0.80, switch_margin=0.25),
        )
        self.assertEqual(outputs[-1], "unknown")

    def test_selection_maximizes_coverage_at_safe_precision(self):
        def candidate(precision, coverage, stability=0.98):
            return {
                "track_final": {"precision": precision, "coverage": coverage},
                "stability": {"transition_stability": stability},
                "config": {"coverage": coverage},
            }

        candidates = [candidate(0.95, 0.20), candidate(0.931, 0.50), candidate(0.92, 0.80)]
        selected, diagnostics = MODULE.select_frontier(candidates)
        self.assertEqual(selected["track_final"]["coverage"], 0.50)
        self.assertTrue(selected["passes"])
        self.assertEqual(diagnostics["passing_count"], 1)


if __name__ == "__main__":
    unittest.main()
