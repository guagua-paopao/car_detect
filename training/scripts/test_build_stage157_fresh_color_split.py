import random
import unittest

from build_stage157_fresh_color_split import HammingIndex, group_for, select_single_color_groups


class Stage157SplitTests(unittest.TestCase):
    def test_hamming_index_matches_bruteforce(self):
        rng = random.Random(157)
        values = [rng.getrandbits(64) for _ in range(500)]
        index = HammingIndex()
        for position, value in enumerate(values):
            index.add(value, f"g{position}")
        for _ in range(200):
            query = rng.getrandbits(64)
            for radius in range(5):
                expected = [i for i, value in enumerate(values) if (query ^ value).bit_count() <= radius]
                self.assertEqual(index.matches(query, radius), expected)

    def test_group_selection_is_group_safe_and_deterministic(self):
        rows = []
        for color in ("black", "white", "brown"):
            for group_index in range(4):
                for row_index in range(3):
                    rows.append({
                        "track_group": f"{color}-{group_index}",
                        "color": color,
                        "color_supervised": "true",
                    })
        first, counts = select_single_color_groups(rows, "dvm", 5, "seed")
        second, _ = select_single_color_groups(rows, "dvm", 5, "seed")
        self.assertEqual(first, second)
        self.assertGreaterEqual(counts["black"], 5)
        for row in rows:
            group = group_for(row, "dvm")
            self.assertEqual(group in first, any(group_for(other, "dvm") in first for other in rows if other["track_group"] == row["track_group"]))


if __name__ == "__main__":
    unittest.main()
