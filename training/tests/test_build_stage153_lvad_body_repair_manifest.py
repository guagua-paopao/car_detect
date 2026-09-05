from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage153_lvad_body_repair_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_stage153_lvad_body_repair_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage153ManifestTests(unittest.TestCase):
    def test_diverse_take_round_robins_groups(self) -> None:
        rows = [
            {"image_path": f"a{i}", "sha256": f"a{i}", "track_group": "a"}
            for i in range(4)
        ] + [
            {"image_path": f"b{i}", "sha256": f"b{i}", "track_group": "b"}
            for i in range(4)
        ]
        picked = MODULE.diverse_take(rows, 2, "seed")
        self.assertEqual({row["track_group"] for row in picked}, {"a", "b"})

    def test_hamming_tree_radius_is_exact(self) -> None:
        tree = MODULE.HammingTree()
        tree.add(0)
        self.assertTrue(tree.contains_within(3, 2))
        self.assertFalse(tree.contains_within(7, 2))

    def test_unknown_coarse_row_is_not_exact(self) -> None:
        row = {"body_type": "unknown", "body_type_supervised": "false", "coarse_body_family": "car"}
        self.assertFalse(MODULE.is_exact_body(row))

    def test_mixed_truck_bus_is_not_admitted_to_truck_loss(self) -> None:
        self.assertTrue(MODULE.admissible_night_family({
            "coarse_body_family": "car", "source_label": "mixed_car_family"
        }))
        self.assertFalse(MODULE.admissible_night_family({
            "coarse_body_family": "truck", "source_label": "mixed_truck_bus_family"
        }))

    def test_exact_replay_clears_redundant_coarse_family(self) -> None:
        prepared = MODULE.prepare_exact_replay({
            "body_type": "heavy_truck",
            "body_type_supervised": "true",
            "coarse_body_family": "truck",
        })
        self.assertEqual(prepared["coarse_body_family"], "")
        self.assertEqual(prepared["stage153_source_coarse_body_family"], "truck")

    def test_scene_flags(self) -> None:
        self.assertTrue(MODULE.is_night({"lighting": "official_night_low_light"}))
        self.assertTrue(MODULE.is_small({"small_target": "true"}))
        self.assertTrue(MODULE.is_occluded_or_truncated({"occlusion_level": "partial"}))


if __name__ == "__main__":
    unittest.main()
