from pathlib import Path
import random
import tempfile
import unittest

import build_stage156_fullscale_body_repair_manifest as builder


class Stage156BuilderTests(unittest.TestCase):
    def test_bktree_allows_same_track_and_rejects_cross_group(self):
        tree = builder.BKTree()
        value = int("0123456789abcdef", 16)
        tree.add(value, "train:track-a")
        self.assertFalse(tree.conflict(value ^ 0b1111, 4, "train:track-a"))
        self.assertTrue(tree.conflict(value ^ 0b1111, 4, "train:track-b"))
        self.assertFalse(tree.conflict(value ^ 0b11111, 4, "train:track-b"))

    def test_block_index_matches_bruteforce_radius_zero_through_four(self):
        generator = random.Random(156)
        tree = builder.BKTree()
        reference: dict[int, set[str]] = {}
        for _ in range(500):
            value = generator.getrandbits(64)
            tag = f"track-{generator.randrange(7)}"
            tree.add(value, tag)
            reference.setdefault(value, set()).add(tag)
        for _ in range(200):
            query = generator.getrandbits(64)
            tag = f"track-{generator.randrange(7)}"
            for radius in range(5):
                expected = any(
                    (query ^ value).bit_count() <= radius and any(existing != tag for existing in tags)
                    for value, tags in reference.items()
                )
                self.assertEqual(tree.conflict(query, radius, tag), expected)

    def test_ua_policy_accepts_only_exact_bus_van_or_coarse_car(self):
        self.assertTrue(builder.ua_admissible({"body_type": "van", "body_type_supervised": "true"}))
        self.assertTrue(builder.ua_admissible({"body_type": "bus", "body_type_supervised": "true"}))
        self.assertFalse(builder.ua_admissible({"body_type": "sedan", "body_type_supervised": "true"}))
        self.assertTrue(builder.ua_admissible({"body_type": "unknown", "body_type_supervised": "false", "coarse_body_family": "car"}))
        self.assertFalse(builder.ua_admissible({"body_type": "unknown", "body_type_supervised": "false", "coarse_body_family": "truck"}))

    def test_exact_rows_clear_redundant_coarse_family_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"fixture")
            row = {
                "image_path": str(image), "body_type": "light_truck", "body_type_supervised": "true",
                "coarse_body_family": "truck", "sample_weight": "1.0",
            }
            prepared = builder.prepare(row, Path(directory) / "manifest.csv", "fixture", 1.0)
            self.assertEqual(prepared["coarse_body_family"], "")
            self.assertEqual(prepared["stage156_source_coarse_body_family"], "truck")
            self.assertEqual(prepared["sample_weight"], "1.600000")


if __name__ == "__main__":
    unittest.main()
