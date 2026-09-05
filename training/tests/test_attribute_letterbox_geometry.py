from __future__ import annotations

import unittest

from PIL import Image
import torch
from torchvision import transforms

from analyze_stage111_body_router_errors import checkpoint_transform
from evaluate_v2_decoupled_shared_validation import SquarePad as EvaluationSquarePad
from train_attribute import SquarePad as TrainingSquarePad


class AttributeLetterboxGeometryTests(unittest.TestCase):
    def test_landscape_padding_is_centered_and_geometry_is_unchanged(self) -> None:
        image = Image.new("RGB", (5, 3), (250, 10, 20))
        output = TrainingSquarePad()(image)
        self.assertEqual(output.size, (5, 5))
        self.assertEqual(output.getpixel((0, 0)), (114, 114, 114))
        self.assertEqual(output.getpixel((2, 1)), (250, 10, 20))
        self.assertEqual(output.getpixel((2, 3)), (250, 10, 20))

    def test_portrait_padding_handles_odd_border_deterministically(self) -> None:
        image = Image.new("RGB", (2, 5), (1, 2, 3))
        output = TrainingSquarePad()(image)
        self.assertEqual(output.size, (5, 5))
        self.assertEqual(output.getpixel((0, 2)), (114, 114, 114))
        self.assertEqual(output.getpixel((1, 2)), (1, 2, 3))
        self.assertEqual(output.getpixel((2, 2)), (1, 2, 3))
        self.assertEqual(output.getpixel((4, 2)), (114, 114, 114))

    def test_training_and_evaluation_padding_are_byte_identical(self) -> None:
        image = Image.new("RGB", (7, 4), (40, 80, 120))
        training = TrainingSquarePad()(image)
        evaluation = EvaluationSquarePad()(image)
        self.assertEqual(training.size, evaluation.size)
        self.assertEqual(training.tobytes(), evaluation.tobytes())

    def test_router_diagnostic_honors_specialist_letterbox_geometry(self) -> None:
        image = Image.new("RGB", (9, 3), (40, 80, 120))
        diagnostic = checkpoint_transform(
            {"input_size": 16, "resize_mode": "letterbox"}, transforms
        )(image)
        expected = transforms.Compose(
            [
                EvaluationSquarePad(),
                transforms.Resize((16, 16), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )(image)
        stretched = checkpoint_transform(
            {"input_size": 16, "resize_mode": "stretch"}, transforms
        )(image)
        self.assertTrue(torch.equal(diagnostic, expected))
        self.assertFalse(torch.equal(diagnostic, stretched))


if __name__ == "__main__":
    unittest.main()
