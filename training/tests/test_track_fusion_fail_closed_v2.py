from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "track_fusion_fail_closed_v2.py"
SPEC = importlib.util.spec_from_file_location("track_fusion_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def value(label: str, confidence: float = 0.95, quality: float = 0.9) -> dict:
    return {"label": label, "confidence": confidence, "quality": quality}


class FailClosedTrackFusionTest(unittest.TestCase):
    def fuse(self, values: list[dict], **overrides) -> list[str]:
        settings = {
            "window": 3,
            "minimum_share": 0.55,
            "minimum_margin": 0.10,
            "switch_confirmations": 3,
            "conflict_unknown_after": 2,
            "override_ratio": 1.15,
        }
        settings.update(overrides)
        return MODULE.fuse_sequence_fail_closed(values, **settings)

    def test_consistent_track_remains_stable(self) -> None:
        self.assertEqual(self.fuse([value("sedan")] * 5), ["sedan"] * 5)

    def test_single_low_quality_conflict_cannot_overwrite_history(self) -> None:
        sequence = [value("suv"), value("suv"), value("sedan", quality=0.05), value("suv")]
        self.assertEqual(self.fuse(sequence), ["suv", "suv", "suv", "suv"])

    def test_continuous_strong_conflict_becomes_unknown_before_switch(self) -> None:
        sequence = [value("suv"), value("sedan"), value("sedan"), value("sedan"), value("sedan")]
        result = self.fuse(sequence, window=1)
        self.assertEqual(result, ["suv", "suv", "unknown", "sedan", "sedan"])

    def test_ambiguous_conflict_fails_closed(self) -> None:
        sequence = [value("suv"), value("sedan"), value("suv"), value("sedan")]
        result = self.fuse(sequence, minimum_share=0.70, minimum_margin=0.40)
        self.assertEqual(result[-1], "unknown")

    def test_unknown_frames_do_not_invent_a_label(self) -> None:
        self.assertEqual(self.fuse([value("unknown"), value("unknown")]), ["unknown", "unknown"])

    def test_invalid_safety_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.fuse([value("suv")], window=0)
        with self.assertRaises(ValueError):
            self.fuse([value("suv")], override_ratio=0.9)


if __name__ == "__main__":
    unittest.main()
