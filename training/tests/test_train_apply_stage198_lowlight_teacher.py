import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage198", ROOT / "scripts" / "train_apply_stage198_lowlight_teacher.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["stage198"] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage198Test(unittest.TestCase):
    def test_threshold_prefers_maximum_recall_at_precision_floor(self):
        threshold, metrics = MODULE.select_precision_threshold(
            [0.99, 0.95, 0.90, 0.80, 0.10], [1, 1, 0, 1, 0], 0.75, 2,
        )
        self.assertEqual(threshold, 0.8)
        self.assertEqual(metrics["precision"], 0.75)
        self.assertEqual(metrics["recall"], 1.0)

    def test_group_split_never_crosses_groups(self):
        rows = []
        for source, label in (("a", 0), ("a", 1), ("b", 0), ("b", 1)):
            for group in range(10):
                rows.append({"source_dataset": source, "scene_target": str(label), "stage177_scene_group_key": f"{source}-{label}-{group}"})
        train, validation, conflicts = MODULE.split_groups(rows, 0.2)
        self.assertEqual(conflicts, 0)
        self.assertFalse({MODULE.group_key(row) for row in train} & {MODULE.group_key(row) for row in validation})
        self.assertEqual({row["scene_target"] for row in train}, {"0", "1"})
        self.assertEqual({row["scene_target"] for row in validation}, {"0", "1"})


if __name__ == "__main__":
    unittest.main()
