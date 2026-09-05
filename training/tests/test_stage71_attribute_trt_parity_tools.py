from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("stage71_trt_prepare", "prepare_stage71_attribute_trt_parity.py")
COMPARE = load("stage71_trt_compare", "compare_stage71_attribute_trt_outputs.py")


class Stage71AttributeTrtParityToolTests(unittest.TestCase):
    def test_evenly_spaced_selection_includes_ends(self) -> None:
        rows = [{"id": str(index)} for index in range(17)]
        selected = PREPARE.evenly_spaced(rows, 8)
        self.assertEqual(selected[0]["id"], "0")
        self.assertEqual(selected[-1]["id"], "16")
        self.assertEqual(len({item["id"] for item in selected}), 8)

    def test_evenly_spaced_rejects_short_pool(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "need 8"):
            PREPARE.evenly_spaced([{"id": "0"}], 8)

    def test_parse_trtexec_preserves_output_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "name": "body_type",
                            "dimensions": "2x3",
                            "values": [0, 1, 2, 3, 4, 5],
                        },
                        {
                            "name": "color",
                            "dimensions": "2x2",
                            "values": [1, 2, 3, 4],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            outputs = COMPARE.parse_trtexec(path)
            self.assertEqual(outputs["body_type"].shape, (2, 3))
            self.assertEqual(outputs["color"].shape, (2, 2))

    def test_softmax_is_normalized(self) -> None:
        values = COMPARE.softmax(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        self.assertAlmostEqual(float(values.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
