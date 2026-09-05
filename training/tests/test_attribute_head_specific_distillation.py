from __future__ import annotations

import argparse
import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = Path(os.environ.get(
    "TRAIN_ATTRIBUTE_SCRIPT",
    Path(__file__).resolve().parents[1] / "scripts" / "train_attribute.py",
))
SPEC = importlib.util.spec_from_file_location("train_attribute_head_teachers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def arguments(**overrides):
    values = {
        "teacher_checkpoint": None,
        "body_teacher_checkpoint": None,
        "color_teacher_checkpoint": None,
        "distill_weight": 0.0,
        "distill_body_weight": 1.0,
        "distill_color_weight": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class HeadSpecificDistillationTests(unittest.TestCase):
    def test_two_specialist_teachers_are_valid(self) -> None:
        mode = MODULE.validate_teacher_configuration(arguments(
            body_teacher_checkpoint=Path("body.pt"),
            color_teacher_checkpoint=Path("color.pt"),
            distill_weight=0.1,
        ))
        self.assertEqual(mode, "head_specific")

    def test_color_only_teacher_must_disable_body_distillation(self) -> None:
        with self.assertRaisesRegex(ValueError, "body-weight 0"):
            MODULE.validate_teacher_configuration(arguments(
                color_teacher_checkpoint=Path("color.pt"),
                distill_weight=0.1,
            ))

    def test_legacy_and_specialist_teachers_cannot_mix(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            MODULE.validate_teacher_configuration(arguments(
                teacher_checkpoint=Path("joint.pt"),
                body_teacher_checkpoint=Path("body.pt"),
                distill_weight=0.1,
            ))

    def test_each_logit_head_comes_from_its_specialist(self) -> None:
        class Teacher:
            def __init__(self, body, color):
                self.outputs = (body, color)

            def __call__(self, _images):
                return self.outputs

        body, color = MODULE.forward_teacher_logits(
            "images",
            body_teacher=Teacher("body-correct", "body-wrong-color"),
            color_teacher=Teacher("color-wrong-body", "color-correct"),
        )
        self.assertEqual(body, "body-correct")
        self.assertEqual(color, "color-correct")


if __name__ == "__main__":
    unittest.main()
