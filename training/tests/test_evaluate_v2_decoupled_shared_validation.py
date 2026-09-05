from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import csv
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_v2_decoupled_shared_validation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v2_shared", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SharedValidationTests(unittest.TestCase):
    def test_training_root_is_importable_for_hierarchy_decoder(self) -> None:
        self.assertIn(str(MODULE.TRAINING_ROOT), sys.path)

    def test_color_aggregation_sums_gray_and_silver_only(self) -> None:
        labels = ["black", "white", "gray", "silver", "red", "blue", "green", "yellow", "brown", "other", "unknown"]
        values = [0.01, 0.02, 0.20, 0.35, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        result = MODULE.aggregate_color_probabilities(labels, values)
        self.assertAlmostEqual(result["silver_gray"], 0.55)
        self.assertAlmostEqual(result["yellow_orange"], 0.06)
        self.assertAlmostEqual(result["brown_beige"], 0.07)
        self.assertNotIn("gray", result)

    def test_v1_baseline_color_aggregation_is_identity(self) -> None:
        labels = list(MODULE.SHARED_COLORS)
        values = [index / 100 for index in range(1, len(labels) + 1)]
        result = MODULE.aggregate_color_probabilities(labels, values)
        self.assertEqual(list(result), labels)
        self.assertEqual(list(result.values()), values)

    def test_production_baseline_taxonomy_is_fail_closed(self) -> None:
        checkpoint = {
            "labels_version": MODULE.PRODUCTION_V1_LABELS["labels_version"],
            "body_types": list(MODULE.PRODUCTION_V1_LABELS["body_types"]),
            "colors": list(MODULE.PRODUCTION_V1_LABELS["colors"]),
        }
        MODULE.validate_checkpoint_taxonomy(checkpoint, checkpoint, None)
        bad_checkpoint = dict(checkpoint)
        bad_checkpoint["colors"] = list(reversed(checkpoint["colors"]))
        with self.assertRaisesRegex(RuntimeError, "production baseline checkpoint taxonomy mismatch"):
            MODULE.validate_checkpoint_taxonomy(checkpoint, bad_checkpoint, None)

    def test_candidate_requires_checkpoint_pinned_truck_hierarchy(self) -> None:
        labels = {
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": [
                "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
                "light_truck", "heavy_truck", "other", "unknown",
            ],
            "colors": [
                "black", "white", "gray", "silver", "red", "blue", "green",
                "yellow", "brown", "other", "unknown",
            ],
        }
        body = {
            **labels,
            "body_hierarchy": "truck_family",
            "truck_subtype_threshold": 0.65,
        }
        color = dict(labels)
        MODULE.validate_checkpoint_taxonomy(body, color, labels)
        bad_body = dict(body)
        bad_body["body_hierarchy"] = "none"
        with self.assertRaisesRegex(RuntimeError, "lost truck-family hierarchy"):
            MODULE.validate_checkpoint_taxonomy(bad_body, color, labels)

    def test_truck_specialist_taxonomy_is_fail_closed(self) -> None:
        checkpoint = {
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["light_truck", "heavy_truck", "unknown"],
            "colors": ["unknown"],
        }
        MODULE.validate_truck_specialist_checkpoint(checkpoint)
        checkpoint["body_types"] = ["truck", "heavy_truck", "unknown"]
        with self.assertRaisesRegex(RuntimeError, "body taxonomy mismatch"):
            MODULE.validate_truck_specialist_checkpoint(checkpoint)

    def test_truck_family_routes_to_confident_exact_specialist(self) -> None:
        labels = [
            "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
            "light_truck", "heavy_truck", "other", "unknown",
        ]
        full = torch.full((1, len(labels)), -4.0)
        full[0, labels.index("truck")] = 2.0
        full[0, labels.index("light_truck")] = 1.0
        full[0, labels.index("heavy_truck")] = 1.0
        specialist = torch.tensor([[-2.0, 4.0, -3.0]])
        prediction, confidence = MODULE.route_body_with_specialist(
            full, labels, specialist,
            ["light_truck", "heavy_truck", "unknown"], 0.80,
        )
        self.assertEqual(labels[int(prediction[0])], "heavy_truck")
        self.assertGreater(float(confidence[0]), 0.80)

    def test_uncertain_truck_subtype_abstains_in_exact_gate(self) -> None:
        labels = [
            "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
            "light_truck", "heavy_truck", "other", "unknown",
        ]
        full = torch.full((1, len(labels)), -4.0)
        full[0, labels.index("truck")] = 3.0
        specialist = torch.tensor([[0.2, 0.1, 1.5]])
        prediction, _ = MODULE.route_body_with_specialist(
            full, labels, specialist,
            ["light_truck", "heavy_truck", "unknown"], 0.80,
        )
        self.assertEqual(labels[int(prediction[0])], "unknown")

    def test_nontruck_family_bypasses_specialist(self) -> None:
        labels = [
            "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
            "light_truck", "heavy_truck", "other", "unknown",
        ]
        full = torch.full((1, len(labels)), -5.0)
        full[0, labels.index("sedan")] = 5.0
        specialist = torch.tensor([[-3.0, 5.0, -3.0]])
        prediction, _ = MODULE.route_body_with_specialist(
            full, labels, specialist,
            ["light_truck", "heavy_truck", "unknown"], 0.80,
        )
        self.assertEqual(labels[int(prediction[0])], "sedan")

    def test_threshold_selection_maximizes_coverage_at_precision_gate(self) -> None:
        predictions = ["sedan", "suv", "sedan", "unknown"]
        confidences = [0.95, 0.85, 0.65, 0.99]
        truths = ["sedan", "suv", "suv", "sedan"]
        selected = MODULE.select_threshold(predictions, confidences, truths, [0.5, 0.7, 0.9], precision_gate=0.93)
        self.assertTrue(selected["precision_gate_found"])
        self.assertEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["selected"], 2)
        self.assertEqual(selected["precision"], 1.0)

    def test_complex_subset_detection_is_explicit(self) -> None:
        self.assertTrue(MODULE.is_complex_row({"vehicle_size": "small"}))
        self.assertTrue(MODULE.is_complex_row({"night": "true"}))
        self.assertTrue(MODULE.is_complex_row({"lighting": "strong_backlight"}))
        self.assertTrue(MODULE.is_complex_row({"weather": "rain"}))
        self.assertTrue(MODULE.is_complex_row({"occlusion_level": "partial"}))
        self.assertFalse(MODULE.is_complex_row({"vehicle_size": "large", "weather": "clear"}))

    def test_per_class_metrics_separate_precision_recall_and_coverage(self) -> None:
        result = MODULE.per_class_metrics(
            ["sedan", "suv", "unknown", "sedan"],
            [0.95, 0.90, 0.99, 0.80],
            ["sedan", "sedan", "suv", "suv"],
            0.85,
        )
        self.assertEqual(result["sedan"]["support"], 2)
        self.assertEqual(result["sedan"]["precision"], 1.0)
        self.assertEqual(result["sedan"]["recall"], 0.5)
        self.assertEqual(result["sedan"]["coverage"], 1.0)
        self.assertEqual(result["suv"]["precision"], 0.0)
        self.assertEqual(result["suv"]["coverage"], 0.0)

    def test_generic_truck_is_family_only_not_exact(self) -> None:
        self.assertFalse(MODULE.body_correct("truck", "light_truck", family_aware=False))
        self.assertTrue(MODULE.body_correct("truck", "light_truck", family_aware=True))

    def test_quality_weighted_fusion_conflict_abstains(self) -> None:
        observations = [
            ("sedan", 0.95, 1.0),
            ("sedan", 0.90, 1.0),
            ("suv", 0.99, 1.0),
            ("suv", 0.99, 1.0),
        ]
        fused = MODULE.fuse_observations(observations, window=5, minimum_share=0.60, minimum_margin=0.15)
        self.assertEqual(fused[-1], "unknown")

    def test_low_quality_conflict_does_not_replace_history(self) -> None:
        observations = [
            ("sedan", 0.95, 1.0),
            ("sedan", 0.90, 1.0),
            ("suv", 0.99, 0.10),
        ]
        fused = MODULE.fuse_observations(observations)
        self.assertEqual(fused[-1], "sedan")

    def test_manifest_rejects_non_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "split", "review_status"])
                writer.writeheader()
                writer.writerow({"image_path": "x.jpg", "split": "test", "review_status": "approved"})
            with self.assertRaisesRegex(RuntimeError, "validation-only manifest"):
                MODULE.resolve_rows(manifest, root)

    def test_manifest_rejects_path_outside_safety_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory:
            root = Path(directory)
            outside = Path(outside_directory) / "x.jpg"
            outside.write_bytes(b"x")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "split", "review_status"])
                writer.writeheader()
                writer.writerow({"image_path": str(outside), "split": "validation", "review_status": "approved"})
            with self.assertRaisesRegex(RuntimeError, "escapes datasets safety root"):
                MODULE.resolve_rows(manifest, root)


if __name__ == "__main__":
    unittest.main()
