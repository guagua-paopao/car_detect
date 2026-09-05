from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "sweep_attribute_track_fusion_fail_closed_v2.py"
SPEC = importlib.util.spec_from_file_location("sweep_track_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def value(label: str, confidence: float = 0.95, quality: float = 0.9) -> dict:
    return {"label": label, "confidence": confidence, "quality": quality}


class FailClosedSweepWrapperTest(unittest.TestCase):
    def test_rejects_test_split_before_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation-only"):
            MODULE.enforce_validation_only(
                ["--split", "test", "--output", "unused.json"]
            )

    def test_requires_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "--output"):
            MODULE.enforce_validation_only(["--split", "validation"])

    def test_adapter_fails_closed_before_same_label_switch(self) -> None:
        sequence = [value("suv"), value("sedan"), value("sedan"), value("sedan")]
        self.assertEqual(
            MODULE.fail_closed_fuse(sequence, 1, 0.55, 0.10),
            ["suv", "suv", "unknown", "sedan"],
        )

    def test_adapter_ignores_low_quality_reverse_frame(self) -> None:
        sequence = [value("suv"), value("suv"), value("sedan", quality=0.01)]
        self.assertEqual(
            MODULE.fail_closed_fuse(sequence, 3, 0.55, 0.10),
            ["suv", "suv", "suv"],
        )


if __name__ == "__main__":
    unittest.main()
