from __future__ import annotations

import unittest

from training.scripts.evaluate_stage73_color_pseudolabel_rule import (
    aggregate_teachers,
    aggregate_views,
    evaluate_configuration,
    foreground_compatible,
    passes_gate,
    wilson_lower,
)


class Stage73ColorRuleTest(unittest.TestCase):
    def test_view_and_teacher_consensus_fail_closed(self) -> None:
        views = [("red", 0.91), ("red", 0.88), ("blue", 0.99)]
        self.assertEqual(aggregate_views(views, 2, 0.85), ("red", 0.88))
        self.assertIsNone(aggregate_views(views, 3, 0.85))
        self.assertEqual(
            aggregate_teachers(
                {"a": views, "b": [("red", 0.93), ("red", 0.90), ("red", 0.89)]},
                2,
                0.85,
            ),
            ("red", 0.88),
        )
        self.assertIsNone(
            aggregate_teachers(
                {"a": views, "b": [("blue", 0.93), ("blue", 0.90), ("blue", 0.89)]},
                2,
                0.85,
            )
        )


    def test_foreground_exact_and_achromatic_compatibility(self) -> None:
        evidence = {
            "label": "unknown",
            "proposed_label": "silver_gray",
            "scores": {
                "black": 0.28,
                "white": 0.20,
                "silver_gray": 0.35,
                "red": 0.02,
                "blue": 0.01,
                "green": 0.00,
                "yellow_orange": 0.00,
                "brown_beige": 0.01,
            },
        }
        self.assertFalse(foreground_compatible(evidence, "silver_gray", "exact_strong", 0.12, 0.25))
        self.assertTrue(foreground_compatible(evidence, "black", "group_compatible", 0.12, 0.25))
        self.assertTrue(foreground_compatible(evidence, "silver_gray", "group_compatible", 0.12, 0.25))
        self.assertFalse(foreground_compatible(evidence, "red", "group_compatible", 0.12, 0.25))


    def test_wilson_and_gate_require_statistical_support(self) -> None:
        self.assertLess(0.93, wilson_lower(150, 150))
        self.assertLess(wilson_lower(150, 150), 1.0)
        passing = {
            "selected": 200,
            "precision": 0.98,
            "precision_wilson_lower_95": 0.95,
            "classes_with_at_least_10_and_precision_0_90": 5,
        }
        self.assertTrue(passes_gate(passing))
        self.assertFalse(passes_gate({**passing, "selected": 149}))
        self.assertFalse(passes_gate({**passing, "precision_wilson_lower_95": 0.92}))

    def test_per_class_gate_uses_predicted_precision(self) -> None:
        rows = [{"color": "red"}, {"color": "blue"}]
        teachers = [
            {"a": [("red", 0.99)], "b": [("red", 0.99)]},
            {"a": [("red", 0.99)], "b": [("red", 0.99)]},
        ]
        foreground = [
            {"label": "red", "proposed_label": "red", "scores": {"red": 1.0}},
            {"label": "red", "proposed_label": "red", "scores": {"red": 1.0}},
        ]
        result = evaluate_configuration(
            rows,
            teachers,
            foreground,
            minimum_views=1,
            confidence_floor=0.90,
            foreground_mode="exact_strong",
            chromatic_floor=0.12,
            achromatic_floor=0.25,
            labels=["red", "blue"],
        )
        self.assertEqual(result["per_predicted_class"]["red"]["selected"], 2)
        self.assertEqual(result["per_predicted_class"]["red"]["precision"], 0.5)
        self.assertEqual(result["per_truth_class"]["blue"]["coverage"], 1.0)
        self.assertEqual(result["per_truth_class"]["blue"]["selected_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
