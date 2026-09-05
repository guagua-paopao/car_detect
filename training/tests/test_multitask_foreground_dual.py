from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import (  # noqa: E402
    ARCHITECTURES,
    MultiTaskVehicleAttributes,
    model_from_checkpoint,
)


class ForegroundDualContractTest(unittest.TestCase):
    def test_foreground_dual_contract(self) -> None:
        model = MultiTaskVehicleAttributes(
            10,
            10,
            architecture="mobilenet_v3_large_foreground_dual",
            pretrained=False,
        ).eval()
        with torch.no_grad():
            for input_size in (224, 256):
                images = torch.zeros(2, 3, input_size, input_size)
                body_features, color_features = model.forward_features(images)
                body, color = model(images)
                self.assertEqual(tuple(body.shape), (2, 10))
                self.assertEqual(tuple(color.shape), (2, 10))
                self.assertEqual(body_features.shape[0], 2)
                self.assertEqual(color_features.shape[0], 2)
                self.assertTrue(torch.equal(body, model.body_type_head(body_features)))
                self.assertTrue(torch.equal(color, model.color_head(color_features)))
        self.assertIsNot(model.body_type_head, model.color_head)

    def test_foreground_dual_checkpoint_reconstruction(self) -> None:
        self.assertIn("mobilenet_v3_large_foreground_dual", ARCHITECTURES)
        checkpoint = {
            "architecture": "mobilenet_v3_large_foreground_dual",
            "body_types": [f"body_{index}" for index in range(10)],
            "colors": [f"color_{index}" for index in range(10)],
        }
        model = model_from_checkpoint(checkpoint, pretrained=False)
        self.assertEqual(model.architecture, "mobilenet_v3_large_foreground_dual")


if __name__ == "__main__":
    unittest.main()
