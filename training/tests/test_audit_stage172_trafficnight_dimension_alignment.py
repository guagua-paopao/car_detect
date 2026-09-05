from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_stage172_trafficnight_dimension_alignment import scale_points  # noqa: E402


class TrafficNightDimensionAlignmentTests(unittest.TestCase):
    def test_identity_transform(self) -> None:
        self.assertEqual(scale_points([[0, 0], [10, 20], [20, 0]], 20, 20, 20, 20), [(0, 0), (10, 20), (20, 0)])

    def test_nonuniform_declared_to_decoded_transform(self) -> None:
        points = scale_points([[0, 0], [4000, 3000], [2000, 1500]], 4000, 3000, 2776, 2208)
        self.assertEqual(points, [(0, 0), (2776, 2208), (1388, 1104)])

    def test_invalid_point_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid point"):
            scale_points([[0, 0], [1], [2, 2]], 2, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
