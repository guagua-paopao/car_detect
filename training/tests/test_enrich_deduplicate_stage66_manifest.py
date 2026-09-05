from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "enrich_deduplicate_stage66_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage66_dedup", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage66DedupTest(unittest.TestCase):
    def test_exact_duplicate_merge_preserves_nonconflicting_truth_and_scene_flags(self) -> None:
        rows = [
            {"image_path": "b.jpg", "body_type": "bus", "body_type_supervised": "true", "color": "unknown", "color_supervised": "false", "night": "false", "sample_weight": "1"},
            {"image_path": "a.jpg", "body_type": "bus", "body_type_supervised": "true", "color": "red", "color_supervised": "true", "night": "true", "sample_weight": "2"},
        ]
        merged, conflicts = MODULE.merge_exact_group(rows)
        self.assertEqual(merged["image_path"], "a.jpg")
        self.assertEqual(merged["body_type"], "bus")
        self.assertEqual(merged["color"], "red")
        self.assertEqual(merged["night"], "true")
        self.assertEqual(merged["sample_weight"], "2.000000")
        self.assertFalse(any(conflicts.values()))

    def test_conflicting_exact_labels_fail_closed_for_that_head(self) -> None:
        rows = [
            {"image_path": "a.jpg", "body_type": "bus", "body_type_supervised": "true", "color": "red", "color_supervised": "true"},
            {"image_path": "b.jpg", "body_type": "van", "body_type_supervised": "true", "color": "red", "color_supervised": "true"},
        ]
        merged, conflicts = MODULE.merge_exact_group(rows)
        self.assertEqual(merged["body_type"], "unknown")
        self.assertEqual(merged["body_type_supervised"], "false")
        self.assertTrue(conflicts["body_type"])
        self.assertEqual(merged["color"], "red")

    def test_near_pair_index_counts_distances(self) -> None:
        count, distances = MODULE.count_near_pairs([0b0000, 0b1111], [0b0001], 4)
        self.assertEqual(count, 2)
        self.assertEqual(distances[1], 1)
        self.assertEqual(distances[3], 1)


if __name__ == "__main__":
    unittest.main()
