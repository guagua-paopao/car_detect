import importlib.util
from pathlib import Path
import sys
import types
import unittest


# The fusion primitive does not need model code.  Stub the sibling evaluator
# while importing so this focused test remains dependency-light.
stub = types.ModuleType("evaluate_stage159_component_validation")
sys.modules.setdefault("evaluate_stage159_component_validation", stub)
SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_stage164_persistent_fusion.py"
SPEC = importlib.util.spec_from_file_location("stage164_fusion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PersistentFusionTests(unittest.TestCase):
    def config(self, **updates):
        value = {
            "window": 5,
            "minimum_observations": 2,
            "entry_share": 0.60,
            "entry_margin": 0.15,
            "switch_share": 0.80,
            "switch_margin": 0.35,
            "switch_patience": 2,
        }
        value.update(updates)
        return value

    def test_low_quality_last_frame_does_not_overwrite_state(self):
        observations = [
            ("suv", 0.96, 1.0),
            ("suv", 0.94, 1.0),
            ("mpv", 0.99, 0.10),
        ]
        self.assertEqual(
            MODULE.persistent_fuse(observations, **self.config()),
            ["unknown", "suv", "suv"],
        )

    def test_repeated_strong_conflict_switches_only_after_patience(self):
        observations = [
            ("suv", 0.95, 1.0),
            ("suv", 0.95, 1.0),
            ("mpv", 0.99, 1.0),
            ("mpv", 0.99, 1.0),
            ("mpv", 0.99, 1.0),
        ]
        outputs = MODULE.persistent_fuse(
            observations,
            **self.config(window=3, switch_share=0.60, switch_margin=0.15),
        )
        self.assertEqual(outputs[:3], ["unknown", "suv", "suv"])
        self.assertEqual(outputs[-1], "mpv")

    def test_ambiguous_conflict_returns_unknown(self):
        observations = [
            ("suv", 0.95, 1.0),
            ("suv", 0.95, 1.0),
            ("mpv", 0.95, 1.0),
            ("mpv", 0.95, 1.0),
        ]
        outputs = MODULE.persistent_fuse(observations, **self.config())
        self.assertEqual(outputs[-1], "unknown")


if __name__ == "__main__":
    unittest.main()
