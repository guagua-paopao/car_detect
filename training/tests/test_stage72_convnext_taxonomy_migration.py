from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from src.multitask_mobilenet_v3 import MultiTaskVehicleAttributes  # noqa: E402


MIGRATOR = TRAINING / "scripts" / "migrate_attribute_checkpoint_taxonomy_v2.py"
LABELS = TRAINING.parent / "config" / "vehicle_labels.v2.json"
V1_BODY = [
    "sedan",
    "suv",
    "mpv",
    "van",
    "pickup",
    "bus",
    "light_truck",
    "heavy_truck",
    "other",
    "unknown",
]
V1_COLOR = [
    "black",
    "white",
    "silver_gray",
    "red",
    "blue",
    "green",
    "yellow_orange",
    "brown_beige",
    "other",
    "unknown",
]


class Stage72ConvnextTaxonomyMigrationTest(unittest.TestCase):
    def test_full_convnext_checkpoint_migrates_shared_backbone_and_fine_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "stage71-color-teacher.pt"
            output_path = root / "stage72-v2-init.pt"
            report_path = root / "report.json"

            model = MultiTaskVehicleAttributes(
                len(V1_BODY), len(V1_COLOR), architecture="convnext_tiny", pretrained=False
            )
            source_state = model.state_dict()
            with torch.no_grad():
                source_state["color_head.weight"][V1_COLOR.index("silver_gray")].fill_(0.375)
                source_state["color_head.bias"][V1_COLOR.index("silver_gray")].fill_(0.125)
                source_state["body_type_head.weight"][V1_BODY.index("light_truck")].fill_(0.2)
                source_state["body_type_head.weight"][V1_BODY.index("heavy_truck")].fill_(0.6)
            sentinel_key = next(key for key in source_state if key.startswith("backbone.features") and key.endswith("weight"))
            sentinel = source_state[sentinel_key].clone()
            torch.save(
                {
                    "model_state": source_state,
                    "architecture": "convnext_tiny_multitask",
                    "input_size": 256,
                    "resize_mode": "stretch",
                    "body_types": V1_BODY,
                    "colors": V1_COLOR,
                },
                source_path,
            )
            del model, source_state

            result = subprocess.run(
                [
                    sys.executable,
                    str(MIGRATOR),
                    "--input-checkpoint",
                    str(source_path),
                    "--labels",
                    str(LABELS),
                    "--output-checkpoint",
                    str(output_path),
                    "--output-report",
                    str(report_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            migrated = torch.load(output_path, map_location="cpu", weights_only=False)
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["architecture"], "convnext_tiny")
            self.assertEqual(migrated["input_size"], 256)
            self.assertTrue(torch.equal(migrated["model_state"][sentinel_key], sentinel))
            gray = migrated["colors"].index("gray")
            silver = migrated["colors"].index("silver")
            self.assertTrue(torch.all(migrated["model_state"]["color_head.weight"][gray] == 0.375))
            self.assertTrue(torch.all(migrated["model_state"]["color_head.weight"][silver] == 0.375))
            self.assertEqual(report["color_row_sources"]["gray"], ["silver_gray"])
            self.assertEqual(report["color_row_sources"]["silver"], ["silver_gray"])
            truck = migrated["body_types"].index("truck")
            self.assertTrue(torch.allclose(
                migrated["model_state"]["body_type_head.weight"][truck],
                torch.full_like(migrated["model_state"]["body_type_head.weight"][truck], 0.4),
            ))
            self.assertFalse(report["frozen_video_used"])
            self.assertFalse(report["production_model_modified"])


if __name__ == "__main__":
    unittest.main()
