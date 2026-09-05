import importlib.util
from pathlib import Path
import sys
import types
import unittest


component_stub = types.ModuleType("evaluate_stage159_component_validation")
stage165_stub = types.ModuleType("evaluate_stage165_track_frontier")
stage165_stub.component = component_stub
sys.modules.setdefault("evaluate_stage165_track_frontier", stage165_stub)
SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_stage166_selective_short_tracks.py"
SPEC = importlib.util.spec_from_file_location("stage166_selective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SelectiveShortTrackTests(unittest.TestCase):
    def test_learned_threshold_never_lowers_base_threshold(self):
        result = MODULE.no_lower_thresholds(
            {"suv": 0.80, "mpv": 0.75},
            {"suv": 0.70, "mpv": 0.90},
        )
        self.assertEqual(result, {"suv": 0.80, "mpv": 0.90})

    def test_missing_class_fails_closed(self):
        result = MODULE.no_lower_thresholds({"suv": 0.80}, {"mpv": 0.90})
        self.assertEqual(result["suv"], 1.01)
        self.assertEqual(result["mpv"], 1.01)


if __name__ == "__main__":
    unittest.main()
