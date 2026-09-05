import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_stage125_bmd_generic_truck_consensus import decide_consensus


class Stage125ConsensusTests(unittest.TestCase):
    def test_accepts_only_exact_teacher_and_view_consensus(self):
        views = [[("light_truck", 0.9)] * 3 for _ in range(3)]
        self.assertEqual(decide_consensus(views, 0.7)[0], "light_truck")
        views[0][2] = ("heavy_truck", 0.9)
        self.assertEqual(decide_consensus(views, 0.7)[1], "augmentation_conflict")

    def test_rejects_low_confidence_and_unknown(self):
        low = [[("heavy_truck", 0.69)] * 3 for _ in range(3)]
        self.assertEqual(decide_consensus(low, 0.7)[1], "confidence_below_threshold")
        unknown = [[("unknown", 0.99)] * 3 for _ in range(3)]
        self.assertEqual(decide_consensus(unknown, 0.7)[1], "unknown_consensus")


if __name__ == "__main__":
    unittest.main()
