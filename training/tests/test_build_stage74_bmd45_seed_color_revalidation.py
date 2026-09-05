from __future__ import annotations

import unittest

from training.scripts.build_stage74_bmd45_seed_color_revalidation import (
    deduplicate_accepts,
    hamming,
)


def item(path: str, label: str, digest: str, dhash: str, confidence: float = 0.9):
    return {
        "row": {"image_path": path, "sha256": digest, "dhash64": dhash},
        "predicted": label,
        "teacher_confidence": confidence,
        "foreground_support": 0.5,
    }


class Stage74BmdSeedTest(unittest.TestCase):
    def test_hamming(self) -> None:
        self.assertEqual(hamming("0000000000000000", "0000000000000003"), 2)

    def test_dedup_removes_exact_and_near_duplicates(self) -> None:
        values = [
            item("a.jpg", "red", "a" * 64, "0000000000000000", 0.99),
            item("b.jpg", "red", "a" * 64, "ffffffffffffffff", 0.98),
            item("c.jpg", "red", "c" * 64, "0000000000000001", 0.97),
            item("d.jpg", "blue", "d" * 64, "0000000000000003", 0.96),
            item("e.jpg", "blue", "e" * 64, "f0f0f0f0f0f0f0f0", 0.95),
        ]
        retained, reasons = deduplicate_accepts(values, maximum_hamming=2)
        self.assertEqual([entry["row"]["image_path"] for entry in retained], ["a.jpg", "e.jpg"])
        self.assertEqual(reasons["exact_sha256"], 1)
        self.assertEqual(reasons["near_duplicate_same_color"], 1)
        self.assertEqual(reasons["near_duplicate_cross_color_conflict"], 1)


if __name__ == "__main__":
    unittest.main()
