import unittest

from evaluate_stage159_component_validation import (
    fine_color_top_label,
    production_truth_label,
    track_metrics,
    validate_color_checkpoint_taxonomy,
)
import evaluate_v2_decoupled_shared_validation as base


class Stage159ComponentTests(unittest.TestCase):
    def test_fine_color_top_label_does_not_merge_v2_taxonomy(self):
        labels = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        self.assertEqual(fine_color_top_label(labels, [0.01, 0.60, 0.20, 0.10, 0.08, 0.01]), ("gray", 0.60))
        self.assertEqual(fine_color_top_label(labels, [0.01, 0.20, 0.60, 0.10, 0.08, 0.01]), ("silver", 0.60))

    def test_track_metrics_uses_track_group_and_quality_fusion(self):
        rows = [
            {"color_supervised": "true", "color": "blue", "track_group": "vehicle-1", "stage159_validation_group": "synthetic-shared", "crop_quality": "good"},
            {"color_supervised": "true", "color": "blue", "track_group": "vehicle-1", "stage159_validation_group": "synthetic-shared", "crop_quality": "good"},
            {"color_supervised": "true", "color": "red", "track_group": "vehicle-2", "stage159_validation_group": "synthetic-shared", "crop_quality": "good"},
        ]
        outputs = [
            {"color_label": "blue", "color_confidence": 0.9},
            {"color_label": "blue", "color_confidence": 0.8},
            {"color_label": "red", "color_confidence": 0.95},
        ]
        result = track_metrics(rows, outputs, "color", {"blue": 0.7, "red": 0.7})
        self.assertEqual(result["track_final"]["evaluated"], 2)
        self.assertEqual(result["track_final"]["precision"], 1.0)
        self.assertEqual(result["track_final"]["coverage"], 1.0)

    def test_production_truth_mapping_preserves_legacy_merged_taxonomy(self):
        self.assertEqual(production_truth_label("color", "gray"), "silver_gray")
        self.assertEqual(production_truth_label("color", "silver"), "silver_gray")
        self.assertEqual(production_truth_label("color", "yellow"), "yellow_orange")
        self.assertEqual(production_truth_label("color", "brown"), "brown_beige")
        self.assertEqual(production_truth_label("color", "blue"), "blue")
        self.assertEqual(production_truth_label("body", "heavy_truck"), "heavy_truck")

    def test_production_color_checkpoint_taxonomy_is_validated(self):
        checkpoint = {
            "labels_version": base.PRODUCTION_V1_LABELS["labels_version"],
            "body_types": list(base.PRODUCTION_V1_LABELS["body_types"]),
            "colors": list(base.PRODUCTION_V1_LABELS["colors"]),
        }
        validate_color_checkpoint_taxonomy(checkpoint, None)
        checkpoint["colors"] = [*checkpoint["colors"][:-2], "unknown", "other"]
        with self.assertRaisesRegex(RuntimeError, "production baseline checkpoint taxonomy mismatch"):
            validate_color_checkpoint_taxonomy(checkpoint, None)

    def test_candidate_structured_hierarchy_metadata_is_normalized(self):
        checkpoint = {
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": [
                "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
                "light_truck", "heavy_truck", "other", "unknown",
            ],
            "colors": [
                "black", "white", "gray", "silver", "red", "blue", "green",
                "yellow", "brown", "other", "unknown",
            ],
            "body_hierarchy": {
                "mode": "truck_family",
                "truck_subtype_threshold": 0.65,
            },
        }
        validate_color_checkpoint_taxonomy(checkpoint, checkpoint)

    def test_color_only_candidate_does_not_require_body_hierarchy(self):
        checkpoint = {
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["unknown"],
            "colors": [
                "black", "white", "gray", "silver", "red", "blue", "green",
                "yellow", "brown", "other", "unknown",
            ],
            "body_hierarchy": {"mode": "none"},
        }
        validate_color_checkpoint_taxonomy(checkpoint, checkpoint)


if __name__ == "__main__":
    unittest.main()
