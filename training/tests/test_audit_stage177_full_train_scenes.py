from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage177_full_train_scenes.py"
SPEC = importlib.util.spec_from_file_location("stage177", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage177Tests(unittest.TestCase):
    def test_explicit_scene_does_not_promote_proxy_to_confirmed_night(self) -> None:
        self.assertEqual(MODULE.explicit_scene({"night": "true", "lighting": "unknown"})[0], "night")
        self.assertEqual(MODULE.explicit_scene({"night": "unknown", "lighting": "low_light"})[0], "low_light")
        self.assertEqual(MODULE.explicit_scene({"night": "unknown", "lighting": "low_light_proxy"})[0], "unknown")

    def test_group_keys_prefer_track_then_video(self) -> None:
        track, scene, bounded = MODULE.group_keys({
            "source_dataset": "s", "camera_id": "c", "video_id": "v", "track_group": "t",
        })
        self.assertTrue(track.endswith("|track|t") and scene == "s|c|v" and bounded)
        video, scene, bounded = MODULE.group_keys({"source_dataset": "s", "video_id": "v"})
        self.assertTrue(video.endswith("|video|v") and scene == "s||v" and not bounded)

    def test_partial_supervision_is_not_a_near_duplicate_conflict(self) -> None:
        self.assertFalse(MODULE.signatures_conflict(("sedan", ""), ("", "red")))
        self.assertTrue(MODULE.signatures_conflict(("sedan", ""), ("suv", "")))

    def test_loader_scope_does_not_depend_on_formal_flag(self) -> None:
        row = {
            "split": "train", "review_status": "approved", "formal_train_eligible": "false",
            "body_type": "sedan", "body_type_supervised": "true",
            "color": "unknown", "color_supervised": "false",
        }
        self.assertTrue(MODULE.eligible_train(row, {"sedan", "suv"}, {"red"}))

    def test_full_inventory_scope_includes_unknown_train_but_never_validation(self) -> None:
        train_unknown = {
            "split": "train", "review_status": "pending",
            "body_type": "unknown", "body_type_supervised": "false",
            "color": "unknown", "color_supervised": "false",
        }
        validation = dict(train_unknown, split="validation")
        self.assertTrue(MODULE.audit_scope(
            train_unknown, {"sedan"}, {"red"}, include_all_train=True,
        ))
        self.assertFalse(MODULE.audit_scope(
            validation, {"sedan"}, {"red"}, include_all_train=True,
        ))

    def test_attribute_supervision_counts_separates_night_truth_from_labels(self) -> None:
        records = [
            {"stage177_scene_label": "night", "body_exact_supervised": True, "color_exact_supervised": False},
            {"stage177_scene_label": "night", "body_exact_supervised": False, "color_exact_supervised": False},
            {"stage177_scene_label": "low_light", "body_exact_supervised": False, "color_exact_supervised": True},
        ]
        counts = MODULE.attribute_supervision_counts(range(3), records)
        self.assertEqual(counts["confirmed_night_rows"], 2)
        self.assertEqual(counts["confirmed_night_exact_body_supervised_rows"], 1)
        self.assertEqual(counts["confirmed_night_exact_color_supervised_rows"], 0)
        self.assertEqual(counts["low_light_exact_color_supervised_rows"], 1)

    def test_exact_duplicate_representative_prefers_reliable_supervision(self) -> None:
        common = {
            "explicit_label": "night", "track_group": "track-1",
            "image_path": "/tmp/same.jpg",
        }
        unlabeled = dict(
            common, row_number=2, body_exact_supervised=False, color_exact_supervised=False,
        )
        labeled = dict(
            common, row_number=3, body_exact_supervised=True, color_exact_supervised=False,
        )
        self.assertLess(
            MODULE.exact_representative_key(labeled),
            MODULE.exact_representative_key(unlabeled),
        )

    def test_grouping_counts_are_source_scoped(self) -> None:
        records = [
            {"source_dataset": "a", "camera_id": "c", "video_id": "v", "track_group": "t", "source_group": "g"},
            {"source_dataset": "a", "camera_id": "c", "video_id": "v", "track_group": "t", "source_group": "g"},
            {"source_dataset": "b", "camera_id": "c", "video_id": "v", "track_group": "t", "source_group": "g"},
        ]
        counts = MODULE.grouping_counts(range(3), records)
        self.assertEqual(counts["distinct_sources"], 2)
        self.assertEqual(counts["distinct_videos"], 2)
        self.assertEqual(counts["distinct_tracks"], 2)
        self.assertEqual(counts["distinct_source_groups"], 2)

    def test_color_loader_can_include_only_coarse_color_rows(self) -> None:
        row = {
            "split": "train", "review_status": "approved",
            "body_type": "unknown", "body_type_supervised": "false",
            "color": "unknown", "color_supervised": "false", "coarse_color_group": "1",
        }
        self.assertFalse(MODULE.eligible_train(row, {"sedan"}, {"red"}, include_coarse_body=False))
        self.assertTrue(MODULE.eligible_train(
            row, {"sedan"}, {"red"}, include_coarse_body=False, include_coarse_color=True,
        ))

    def test_dhash_and_even_spacing_are_deterministic(self) -> None:
        gray = np.tile(np.arange(9, dtype=np.uint8), (8, 1))
        self.assertEqual(MODULE.dhash64(gray), (1 << 64) - 1)
        self.assertEqual(MODULE.evenly_spaced(list(range(10)), 3), {0, 4, 9})


if __name__ == "__main__":
    unittest.main()
