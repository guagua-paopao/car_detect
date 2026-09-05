from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_ROOT / "scripts" / "evaluate_attribute_track_fusion_fail_closed_v2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_track_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixed_arguments(split: str = "test") -> list[str]:
    return [
        "--split", split,
        "--output", "unused.json",
        "--fusion-windows", "5",
        "--fusion-min-share", "0.6",
        "--fusion-min-margin", "0.1",
        "--type-threshold", "0.75",
        "--color-threshold", "0.70",
    ]


class FixedFailClosedTrackEvaluatorTest(unittest.TestCase):
    def test_one_fixed_window_is_accepted_for_test(self) -> None:
        output, split = MODULE.validate_fixed_protocol(fixed_arguments())
        self.assertEqual(output, Path("unused.json"))
        self.assertEqual(split, "test")

    def test_multiple_windows_are_rejected_before_test_inference(self) -> None:
        arguments = fixed_arguments()
        index = arguments.index("--fusion-windows") + 2
        arguments.insert(index, "3")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.validate_fixed_protocol(arguments)

    def test_missing_fixed_margin_is_rejected(self) -> None:
        arguments = fixed_arguments()
        index = arguments.index("--fusion-min-margin")
        del arguments[index:index + 2]
        with self.assertRaisesRegex(ValueError, "fusion-min-margin"):
            MODULE.validate_fixed_protocol(arguments)


if __name__ == "__main__":
    unittest.main()
