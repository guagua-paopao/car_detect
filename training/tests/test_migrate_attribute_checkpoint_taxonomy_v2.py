from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_attribute_checkpoint_taxonomy_v2.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_migrator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TaxonomyCheckpointMigrationTest(unittest.TestCase):
    def test_fine_color_rows_inherit_merged_v1_initialization(self) -> None:
        old = ["black", "silver_gray", "yellow_orange", "brown_beige", "unknown"]
        new = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        source = {
            "color_head.weight": torch.arange(10, dtype=torch.float32).reshape(5, 2),
            "color_head.bias": torch.arange(5, dtype=torch.float32),
        }
        target = {
            "color_head.weight": torch.zeros(6, 2),
            "color_head.bias": torch.zeros(6),
        }
        audit = MODULE.copy_output_rows(source, target, "color_head", old, new, "color")
        self.assertTrue(torch.equal(target["color_head.weight"][1], source["color_head.weight"][1]))
        self.assertTrue(torch.equal(target["color_head.weight"][2], source["color_head.weight"][1]))
        self.assertEqual(audit["gray"], ["silver_gray"])
        self.assertEqual(audit["silver"], ["silver_gray"])

    def test_generic_truck_starts_from_fine_subtype_mean(self) -> None:
        old = ["sedan", "light_truck", "heavy_truck", "unknown"]
        new = ["sedan", "truck", "light_truck", "heavy_truck", "unknown"]
        source = {
            "body_type_head.weight": torch.tensor([[0.0], [2.0], [4.0], [8.0]]),
            "body_type_head.bias": torch.tensor([0.0, 2.0, 4.0, 8.0]),
        }
        target = {
            "body_type_head.weight": torch.zeros(5, 1),
            "body_type_head.bias": torch.zeros(5),
        }
        audit = MODULE.copy_output_rows(source, target, "body_type_head", old, new, "body")
        self.assertEqual(float(target["body_type_head.weight"][1]), 3.0)
        self.assertEqual(audit["truck"], ["light_truck", "heavy_truck"])

    def test_truck_specialist_rows_copy_exact_source_classes(self) -> None:
        old = ["sedan", "truck", "light_truck", "heavy_truck", "unknown"]
        new = ["light_truck", "heavy_truck", "unknown"]
        source = {
            "body_type_head.weight": torch.arange(10, dtype=torch.float32).reshape(5, 2),
            "body_type_head.bias": torch.arange(5, dtype=torch.float32),
        }
        target = {
            "body_type_head.weight": torch.zeros(3, 2),
            "body_type_head.bias": torch.zeros(3),
        }
        audit = MODULE.copy_output_rows(source, target, "body_type_head", old, new, "body")
        self.assertTrue(torch.equal(target["body_type_head.weight"][0], source["body_type_head.weight"][2]))
        self.assertTrue(torch.equal(target["body_type_head.weight"][1], source["body_type_head.weight"][3]))
        self.assertTrue(torch.equal(target["body_type_head.weight"][2], source["body_type_head.weight"][4]))
        self.assertEqual(audit["light_truck"], ["light_truck"])

    def test_ambiguous_output_layer_detection_fails_closed(self) -> None:
        state = {
            "head.0.weight": torch.zeros(3, 2),
            "head.0.bias": torch.zeros(3),
            "head.1.weight": torch.zeros(3, 2),
            "head.1.bias": torch.zeros(3),
        }
        with self.assertRaises(RuntimeError):
            MODULE.output_layer_keys(state, "head", 3)


if __name__ == "__main__":
    unittest.main()
