from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from training.src.multitask_mobilenet_v3 import MultiTaskVehicleAttributes


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_attribute_checkpoint_architecture.py"
SPEC = importlib.util.spec_from_file_location("architecture_migrator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_checkpoint() -> dict:
    model = MultiTaskVehicleAttributes(3, 4, architecture="mobilenet_v3_large", pretrained=False)
    with torch.no_grad():
        first = next(value for key, value in model.state_dict().items() if key.startswith("backbone.") and value.is_floating_point())
        first.fill_(0.125)
    return {
        "architecture": "mobilenet_v3_large_multitask",
        "model_state": model.state_dict(),
        "body_types": ["sedan", "suv", "unknown"],
        "colors": ["black", "white", "red", "unknown"],
        "input_size": 224,
        "resize_mode": "stretch",
    }


class ArchitectureCheckpointMigrationTest(unittest.TestCase):
    def test_copies_entire_backbone_and_leaves_new_dual_heads(self) -> None:
        source = source_checkpoint()
        migrated, audit = MODULE.migrate_checkpoint(source, "mobilenet_v3_large_dual", 17)
        target = migrated["model_state"]
        source_backbone = {k: v for k, v in source["model_state"].items() if k.startswith("backbone.")}
        self.assertEqual(audit["target_backbone_tensor_count"], len(source_backbone))
        self.assertEqual(audit["copied_backbone_tensor_count"], len(source_backbone))
        for key, value in source_backbone.items():
            self.assertTrue(torch.equal(target[key], value), key)
        self.assertIn("body_type_head.0.weight", audit["fresh_head_tensors"])
        self.assertIn("color_head.3.weight", audit["fresh_head_tensors"])

    def test_fresh_heads_are_seed_deterministic(self) -> None:
        source = source_checkpoint()
        first, _ = MODULE.migrate_checkpoint(source, "mobilenet_v3_large_foreground_dual", 31)
        second, _ = MODULE.migrate_checkpoint(source, "mobilenet_v3_large_foreground_dual", 31)
        self.assertTrue(torch.equal(first["model_state"]["color_head.0.weight"], second["model_state"]["color_head.0.weight"]))

    def test_rejects_cross_family_target(self) -> None:
        with self.assertRaises(RuntimeError):
            MODULE.migrate_checkpoint(source_checkpoint(), "convnext_tiny", 1)


if __name__ == "__main__":
    unittest.main()
