import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage167_complex_body_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage167_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComplexManifestTests(unittest.TestCase):
    def test_clear_large_day_sample_is_unchanged(self):
        multiplier, reasons = MODULE.bounded_multiplier({
            "vehicle_size": "large", "lighting": "day", "weather": "clear",
            "occlusion_level": "none",
        })
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(reasons, [])

    def test_combined_complex_weight_is_bounded(self):
        multiplier, reasons = MODULE.bounded_multiplier({
            "vehicle_size": "small", "night": "true", "blur": "true",
            "occluded": "true", "weather": "rain",
        })
        self.assertEqual(multiplier, 2.5)
        self.assertIn("small", reasons)
        self.assertIn("adverse_weather", reasons)

    def test_frozen_markers_are_detected(self):
        self.assertIsNotNone(MODULE.FROZEN_MARKER_RE.search("vcas_rtsp_demo_60s.mp4"))
        self.assertIsNotNone(MODULE.FROZEN_MARKER_RE.search("clip-36-48/frame.jpg"))


if __name__ == "__main__":
    unittest.main()
