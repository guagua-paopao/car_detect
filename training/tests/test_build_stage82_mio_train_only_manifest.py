from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_stage82_mio_train_only_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage82_mio", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage82MioTrainOnlyTests(unittest.TestCase):
    def test_mapping_keeps_single_unit_truck_generic(self) -> None:
        self.assertEqual(MODULE.SOURCE_TO_BODY["single_unit_truck"], "truck")
        self.assertEqual(MODULE.SOURCE_TO_BODY["articulated_truck"], "heavy_truck")
        self.assertNotIn("car", MODULE.SOURCE_TO_BODY)

    def test_inspect_is_deterministic_and_marks_low_luminance_as_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mio_1.jpg"
            Image.new("RGB", (96, 96), (30, 30, 30)).save(path)
            digest1, dhash1, metrics1 = MODULE.inspect(path)
            digest2, dhash2, metrics2 = MODULE.inspect(path)
            self.assertEqual((digest1, dhash1, metrics1), (digest2, dhash2, metrics2))
            self.assertEqual(metrics1["lighting"], "low_luminance_proxy")

    def test_output_contract_contains_train_only_safety_fields(self) -> None:
        self.assertIn("split", MODULE.FIELDS)
        self.assertIn("color_supervised", MODULE.FIELDS)
        self.assertIn("track_group", MODULE.FIELDS)
        self.assertIn("dhash64", MODULE.FIELDS)


if __name__ == "__main__":
    unittest.main()
