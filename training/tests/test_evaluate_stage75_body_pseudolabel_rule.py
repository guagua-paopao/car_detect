from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_stage75_body_pseudolabel_rule.py"
SPEC = importlib.util.spec_from_file_location("stage75_body_rule", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BodyPseudoLabelRuleTests(unittest.TestCase):
    def test_image_path_resolution_supports_manifest_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_parent = root / "validation-set"
            image = manifest_parent / "crops" / "vehicle.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-decoded-in-this-unit-test")
            self.assertEqual(
                MODULE.resolve_image_path("crops/vehicle.jpg", root, manifest_parent),
                image,
            )

    def test_view_aggregation_is_fail_closed_on_conflict_and_low_confidence(self) -> None:
        self.assertIsNone(MODULE.aggregate_views([("sedan", 0.9), ("suv", 0.9)], 2, 0.5))
        self.assertIsNone(MODULE.aggregate_views([("sedan", 0.9), ("sedan", 0.4)], 2, 0.5))
        self.assertEqual(
            MODULE.aggregate_views([("sedan", 0.9), ("sedan", 0.8)], 2, 0.5),
            ("sedan", 0.8),
        )

    def test_teacher_aggregation_requires_every_teacher_to_agree(self) -> None:
        agreeing = {
            "a": [("bus", 0.9), ("bus", 0.8)],
            "b": [("bus", 0.85), ("bus", 0.82)],
            "c": [("bus", 0.88), ("bus", 0.81)],
        }
        self.assertEqual(MODULE.aggregate_teachers(agreeing, 2, 0.8), ("bus", 0.8))
        agreeing["c"] = [("van", 0.9), ("van", 0.9)]
        self.assertIsNone(MODULE.aggregate_teachers(agreeing, 2, 0.8))

    def test_gate_requires_precision_wilson_support_and_class_diversity(self) -> None:
        passing = {
            "selected": 300,
            "precision": 0.98,
            "precision_wilson_lower_95": 0.95,
            "classes_with_at_least_10_and_precision_0_90": 5,
        }
        self.assertTrue(MODULE.passes_gate(passing))
        for key, failing_value in (
            ("selected", 149),
            ("precision", 0.959),
            ("precision_wilson_lower_95", 0.929),
            ("classes_with_at_least_10_and_precision_0_90", 4),
        ):
            candidate = dict(passing)
            candidate[key] = failing_value
            self.assertFalse(MODULE.passes_gate(candidate))


if __name__ == "__main__":
    unittest.main()
