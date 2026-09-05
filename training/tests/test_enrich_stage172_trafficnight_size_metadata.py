from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from enrich_stage172_trafficnight_size_metadata import bbox_metrics  # noqa: E402


class Stage172TrafficNightSizeMetadataTests(unittest.TestCase):
    def test_bbox_ratio_uses_decoded_source_frame(self) -> None:
        row = {
            "stage172_polygon_points": "[[0,0],[100,0],[100,50],[0,50]]",
            "stage172_decoded_dimensions": "1000x500",
        }
        width, height, ratio = bbox_metrics(row)
        self.assertEqual((width, height), (100, 50))
        self.assertAlmostEqual(ratio, 0.01)

    def test_degenerate_polygon_fails_closed(self) -> None:
        row = {
            "stage172_polygon_points": "[[1,1],[1,1],[1,1]]",
            "stage172_decoded_dimensions": "100x100",
        }
        with self.assertRaisesRegex(ValueError, "degenerate"):
            bbox_metrics(row)


if __name__ == "__main__":
    unittest.main()
