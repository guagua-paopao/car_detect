from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.scripts.build_stage73_uadetrac_color_pseudolabels_v2 import (
    choose_track_decision,
    foreground_support,
    load_validation_rule,
    require_manifest_dataset_root,
    select_readable_audit_rows,
    sha256,
)


class Stage73TrainOnlyMinerTest(unittest.TestCase):
    def test_track_decision_counts_abstentions_and_rejects_conflicts(self) -> None:
        self.assertEqual(
            choose_track_decision(["red", "red", "red", "unknown", "unknown"], 3, 0.60),
            ("red", 0.60, "accepted"),
        )
        self.assertEqual(
            choose_track_decision(["red", "red", "red", "blue", "unknown"], 3, 0.60)[2],
            "accepted_frame_conflict",
        )
        self.assertEqual(
            choose_track_decision(["red", "red", "unknown", "unknown", "unknown"], 3, 0.60)[2],
            "insufficient_accepted_frames",
        )

    def test_foreground_support_uses_predicted_class(self) -> None:
        evidence = {"score": 0.9, "scores": {"red": 0.23, "blue": 0.04}}
        self.assertEqual(foreground_support(evidence, "red"), 0.23)

    def test_readable_selection_replaces_stale_sample_paths(self) -> None:
        rows = [{"frame_number": str(index)} for index in range(40)]
        selected, rejected = select_readable_audit_rows(
            rows,
            audit_frames=5,
            minimum_frames=3,
            is_readable=lambda row: int(row["frame_number"]) not in {0, 8, 16},
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(rejected, 3)
        self.assertTrue(all(int(row["frame_number"]) not in {0, 8, 16} for row in selected))

    def test_dataset_root_is_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("image_path\n", encoding="utf-8")
            require_manifest_dataset_root(manifest, root)
            with self.assertRaises(RuntimeError):
                require_manifest_dataset_root(manifest, root / "wrong")

    def test_validation_rule_must_be_passing_and_sha_pinned(self) -> None:
        selected = {
            "selected": 454,
            "precision": 0.989,
            "precision_wilson_lower_95": 0.974,
            "classes_with_at_least_10_and_precision_0_90": 8,
            "parameters": {
                "minimum_views": 2,
                "confidence_floor": 0.70,
                "foreground_mode": "group_compatible",
                "chromatic_floor": 0.12,
                "achromatic_floor": 0.25,
            },
        }
        payload = {
            "status": "pass_validation_rule_available",
            "selected": selected,
            "policy": {
                "usage": "validation_threshold_selection_only",
                "truth_used_as_teacher_input": False,
                "foreground_pixels_only_accept_or_reject_teacher_consensus": True,
                "test_accessed": False,
                "frozen_video_used": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rule.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parameters, actual = load_validation_rule(path, sha256(path))
            self.assertEqual(parameters, selected["parameters"])
            self.assertEqual(actual, sha256(path))
            with self.assertRaises(RuntimeError):
                load_validation_rule(path, "0" * 64)
            payload["policy"]["test_accessed"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_validation_rule(path, sha256(path))


if __name__ == "__main__":
    unittest.main()
