import unittest

from train_attribute import color_sampling_multipliers


class Stage158ColorSamplingTests(unittest.TestCase):
    def test_inverse_sqrt_balances_long_tail_with_a_hard_cap(self):
        rows = (
            [{"color": "white", "color_supervised": "true"}] * 16
            + [{"color": "green", "color_supervised": "true"}] * 4
            + [{"color": "brown", "color_supervised": "true"}]
            + [{"color": "brown", "color_supervised": "false"}] * 20
            + [{"color": "unknown", "color_supervised": "true"}] * 20
        )
        multipliers, counts = color_sampling_multipliers(
            rows, ["white", "green", "brown", "unknown"], "inverse_sqrt", 3.0
        )
        self.assertEqual(counts, {"brown": 1, "green": 4, "white": 16})
        self.assertEqual(multipliers["white"], 1.0)
        self.assertEqual(multipliers["green"], 2.0)
        self.assertEqual(multipliers["brown"], 3.0)

    def test_none_preserves_unit_sampling(self):
        multipliers, counts = color_sampling_multipliers(
            [{"color": "red", "color_supervised": "true"}],
            ["red", "blue", "unknown"],
            "none",
            4.0,
        )
        self.assertEqual(counts, {"blue": 0, "red": 1})
        self.assertEqual(multipliers, {"blue": 1.0, "red": 1.0})

    def test_rejects_invalid_cap(self):
        with self.assertRaises(ValueError):
            color_sampling_multipliers([], ["red", "unknown"], "inverse_sqrt", 0.5)


if __name__ == "__main__":
    unittest.main()
