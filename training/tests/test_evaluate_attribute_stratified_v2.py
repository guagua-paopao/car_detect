from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "evaluate_attribute_stratified_v2.py"
SPEC = importlib.util.spec_from_file_location("attribute_stratified_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AttributeStratifiedV2Test(unittest.TestCase):
    def test_selective_per_class_metrics_include_abstention(self) -> None:
        metrics = MODULE.selective_per_class(
            predictions=[0, 0, 1, 2],
            targets=[0, 1, 1, 0],
            confidences=[0.95, 0.40, 0.90, 0.99],
            labels=["sedan", "suv", "unknown"],
            threshold=0.75,
        )
        self.assertEqual(metrics["sedan"]["truth_support"], 2)
        self.assertEqual(metrics["sedan"]["emitted_as_class"], 1)
        self.assertEqual(metrics["sedan"]["precision"], 1.0)
        self.assertEqual(metrics["sedan"]["coverage"], 0.5)
        self.assertEqual(metrics["suv"]["recall"], 0.5)

    def test_expanded_metadata_includes_required_complex_strata(self) -> None:
        metadata = MODULE.expanded_metadata(
            {"lighting": "night", "vehicle_size": "small"},
            {
                "camera_id": "gate-01",
                "video_id": "night-07",
                "weather": "rain",
                "blur": "true",
                "glare": "1",
                "vehicle_overlap": "yes",
            },
        )
        self.assertEqual(metadata["camera_source"], "gate-01")
        self.assertEqual(metadata["video_source"], "night-07")
        self.assertEqual(metadata["motion_blur"], "blurred")
        self.assertEqual(metadata["reflection"], "reflective")
        self.assertEqual(metadata["vehicle_overlap"], "overlap")

    def test_frozen_video_artifact_is_rejected(self) -> None:
        arguments = [
            "--split", "validation",
            "--output", "unused.json",
            "--manifest", "vcas_rtsp_demo_60s.mp4",
            "--checkpoint", "candidate.pt",
            "--type-threshold", "0.75",
            "--color-threshold", "0.70",
        ]
        with self.assertRaisesRegex(ValueError, "frozen"):
            MODULE.validate_fixed_inputs(arguments)


if __name__ == "__main__":
    unittest.main()
