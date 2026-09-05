import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from reset_stage129_specialist_head import reset_body_head

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from src.multitask_mobilenet_v3 import MultiTaskVehicleAttributes


class ResetSpecialistHeadTests(unittest.TestCase):
    def test_only_body_head_changes(self):
        model = MultiTaskVehicleAttributes(3, 1, architecture="convnext_tiny", pretrained=False)
        checkpoint = {
            "architecture": "convnext_tiny_multitask",
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["light_truck", "heavy_truck", "unknown"],
            "colors": ["unknown"],
            "model_state": model.state_dict(),
        }
        output, report = reset_body_head(checkpoint, 123)
        self.assertTrue(report["unchanged_non_head"])
        self.assertEqual(report["backbone_digest_before"], report["backbone_digest_after"])
        self.assertNotEqual(report["body_head_digest_before"], report["body_head_digest_after"])
        self.assertEqual(output["stage129_reset_seed"], 123)


if __name__ == "__main__":
    unittest.main()
