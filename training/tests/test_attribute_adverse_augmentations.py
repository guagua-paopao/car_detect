from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_attribute.py"
SPEC = importlib.util.spec_from_file_location("train_attribute_augments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_image() -> Image.Image:
    yy, xx = np.mgrid[0:48, 0:64]
    array = np.stack(
        [40 + xx * 2, 60 + yy * 3, 80 + (xx + yy)], axis=2
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class AdverseAugmentationTest(unittest.TestCase):
    def check_contract(self, transform) -> Image.Image:
        random.seed(20260825)
        original = sample_image()
        output = transform(original)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, original.size)
        self.assertEqual(np.asarray(output).dtype, np.uint8)
        return output

    def test_jpeg_contract(self) -> None:
        output = self.check_contract(
            MODULE.RandomJPEGCompression(1.0, minimum_quality=50, maximum_quality=50)
        )
        self.assertFalse(np.array_equal(np.asarray(output), np.asarray(sample_image())))

    def test_color_temperature_contract(self) -> None:
        output = self.check_contract(MODULE.RandomColorTemperature(1.0, maximum_shift=0.08))
        self.assertFalse(np.array_equal(np.asarray(output), np.asarray(sample_image())))

    def test_shadow_highlight_contract(self) -> None:
        output = self.check_contract(MODULE.RandomShadowHighlight(1.0))
        self.assertFalse(np.array_equal(np.asarray(output), np.asarray(sample_image())))

    def test_zero_probability_is_identity(self) -> None:
        image = sample_image()
        self.assertIs(MODULE.RandomJPEGCompression(0.0, 50, 90)(image), image)
        self.assertIs(MODULE.RandomColorTemperature(0.0, 0.08)(image), image)
        self.assertIs(MODULE.RandomShadowHighlight(0.0)(image), image)


if __name__ == "__main__":
    unittest.main()
