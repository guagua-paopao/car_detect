from __future__ import annotations

import unittest
import sys
import importlib.util
from pathlib import Path

import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.attribute_hierarchy import (  # noqa: E402
    car_family_indices,
    coarse_color_group_indices,
    coarse_truck_family_indices,
    decode_truck_family,
    partial_coarse_family_loss,
    partial_label_group_loss,
    partial_truck_family_loss,
    truck_family_match,
    truck_hierarchy,
)

TAXONOMY_SCRIPT = TRAINING_ROOT / "scripts" / "train_attribute_taxonomy_v2.py"
TRAIN_SCRIPT = TAXONOMY_SCRIPT if TAXONOMY_SCRIPT.is_file() else TRAINING_ROOT / "scripts" / "train_attribute.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("hierarchy_train_attribute", TRAIN_SCRIPT)
assert TRAIN_SPEC and TRAIN_SPEC.loader
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)


LABELS = ["sedan", "truck", "light_truck", "heavy_truck", "unknown"]


class AttributeHierarchyTest(unittest.TestCase):
    def test_generic_truth_supervises_family_probability_not_flat_class(self) -> None:
        hierarchy = truck_hierarchy(LABELS)
        logits = torch.tensor([[0.0, -4.0, 3.0, 2.0, -3.0]], requires_grad=True)
        targets = torch.tensor([hierarchy.generic])
        ordinary = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        loss = partial_truck_family_loss(logits, targets, hierarchy, ordinary)
        self.assertLess(float(loss.detach()), float(ordinary.detach()))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_generic_family_loss_supports_class_and_focal_weighting(self) -> None:
        hierarchy = truck_hierarchy(LABELS)
        logits = torch.tensor([[0.0, -1.0, 1.0, 0.5, -2.0]])
        targets = torch.tensor([hierarchy.generic])
        ordinary = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        base = partial_truck_family_loss(logits, targets, hierarchy, ordinary)
        weighted = partial_truck_family_loss(
            logits,
            targets,
            hierarchy,
            ordinary,
            coarse_class_weight=torch.tensor(2.0),
        )
        focal = partial_truck_family_loss(logits, targets, hierarchy, ordinary, gamma=2.0)
        self.assertAlmostEqual(float(weighted), float(base) * 2.0, places=5)
        self.assertLess(float(focal), float(base))

    def test_coarse_car_truth_supervises_only_family_probability_mass(self) -> None:
        labels = ["sedan", "suv", "mpv", "other", "truck", "unknown"]
        family = car_family_indices(labels)
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, 3.0, -2.0]], requires_grad=True)
        loss = partial_coarse_family_loss(logits, family, gamma=1.0)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_coarse_car_family_requires_deployable_fine_labels(self) -> None:
        with self.assertRaises(ValueError):
            car_family_indices(["sedan", "suv", "unknown"])

    def test_coarse_truck_family_supports_v1_without_generic_output(self) -> None:
        labels = ["sedan", "light_truck", "heavy_truck", "unknown"]
        family = coarse_truck_family_indices(labels)
        self.assertEqual(family, (1, 2))
        logits = torch.tensor([[0.0, 2.0, 1.0, -1.0]], requires_grad=True)
        loss = partial_coarse_family_loss(logits, family)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_coarse_truck_family_includes_v2_generic_output(self) -> None:
        labels = ["truck", "light_truck", "heavy_truck", "unknown"]
        self.assertEqual(coarse_truck_family_indices(labels), (0, 1, 2))

    def test_legacy_color_group_supervises_probability_mass_without_fabrication(self) -> None:
        labels = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        groups = coarse_color_group_indices(labels)
        logits = torch.tensor([[0.0, 2.0, 1.5, -1.0, -1.0, -2.0]], requires_grad=True)
        targets = torch.tensor([1])
        loss = partial_label_group_loss(logits, targets, groups)
        exact_gray = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        self.assertLess(float(loss.detach()), float(exact_gray.detach()))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_partial_color_group_loss_supports_mixed_groups_and_weights(self) -> None:
        labels = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        groups = coarse_color_group_indices(labels)
        logits = torch.tensor([[0.0, 2.0, 1.0, -1.0, -1.0, -2.0], [0.0, -1.0, -1.0, 2.0, 0.0, -2.0]])
        targets = torch.tensor([1, 2])
        loss = partial_label_group_loss(
            logits, targets, groups, sample_weights=torch.tensor([2.0, 1.0]), gamma=1.0
        )
        self.assertGreater(float(loss), 0.0)

    def test_partial_color_group_loss_promotes_half_logits_for_amp_weights(self) -> None:
        labels = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        groups = coarse_color_group_indices(labels)
        logits = torch.tensor(
            [[0.0, 2.0, 1.0, -1.0, -1.0, -2.0]],
            dtype=torch.float16,
            requires_grad=True,
        )
        loss = partial_label_group_loss(
            logits,
            torch.tensor([1]),
            groups,
            sample_weights=torch.tensor([0.5], dtype=torch.float32),
            gamma=1.5,
        )
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_partial_color_group_loss_rejects_unknown_group_code(self) -> None:
        labels = ["black", "gray", "silver", "yellow", "brown", "unknown"]
        groups = coarse_color_group_indices(labels)
        with self.assertRaises(ValueError):
            partial_label_group_loss(torch.zeros(1, len(labels)), torch.tensor([9]), groups)

    def test_decoder_returns_coarse_truck_when_subtype_is_uncertain(self) -> None:
        hierarchy = truck_hierarchy(LABELS)
        logits = torch.tensor([[0.0, 0.1, 1.0, 0.9, -2.0]])
        prediction, confidence = decode_truck_family(logits, hierarchy, 0.65)
        self.assertEqual(int(prediction.item()), hierarchy.generic)
        self.assertGreater(float(confidence.item()), 0.5)

    def test_decoder_returns_fine_subtype_when_conditional_probability_is_clear(self) -> None:
        hierarchy = truck_hierarchy(LABELS)
        logits = torch.tensor([[0.0, -2.0, 4.0, 0.0, -2.0]])
        prediction, _ = decode_truck_family(logits, hierarchy, 0.65)
        self.assertEqual(int(prediction.item()), hierarchy.light)

    def test_coarse_match_does_not_hide_wrong_fine_subtype(self) -> None:
        hierarchy = truck_hierarchy(LABELS)
        self.assertTrue(truck_family_match(hierarchy.generic, hierarchy.heavy, hierarchy))
        self.assertTrue(truck_family_match(hierarchy.light, hierarchy.generic, hierarchy))
        self.assertFalse(truck_family_match(hierarchy.light, hierarchy.heavy, hierarchy))

    def test_missing_hierarchy_label_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            truck_hierarchy(["truck", "heavy_truck"])

    def test_evaluation_keeps_exact_and_family_metrics_separate(self) -> None:
        hierarchy = truck_hierarchy(LABELS)

        class FixedModel(torch.nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                body = torch.tensor([[0.0, 0.1, 1.0, 0.9, -2.0]]).repeat(batch, 1)
                color = torch.tensor([[4.0, 0.0]]).repeat(batch, 1)
                return body, color

        loader = [
            (
                torch.zeros(1, 3, 8, 8),
                torch.tensor([hierarchy.heavy]),
                torch.tensor([0]),
                [{}],
            )
        ]
        metrics = TRAIN_MODULE.evaluate(
            FixedModel(),
            loader,
            torch.device("cpu"),
            len(LABELS),
            2,
            LABELS.index("unknown"),
            1,
            0.5,
            0.5,
            hierarchy,
            0.65,
        )
        self.assertEqual(metrics["body_type_accuracy"], 0.0)
        self.assertEqual(metrics["body_type_family_accuracy"], 1.0)
        self.assertEqual(metrics["body_type_high_confidence_precision"], 0.0)
        self.assertEqual(metrics["body_type_family_high_confidence_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
