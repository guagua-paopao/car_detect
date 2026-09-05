from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest

import build_stage155_new_unused_holdout as builder


class Stage155HoldoutTests(unittest.TestCase):
    def test_bktree_allows_same_track_but_rejects_cross_group(self):
        tree = builder.BKTree()
        value = int("0123456789abcdef", 16)
        tree.add(value, "track-a")
        self.assertFalse(tree.conflict(value, 4, "track-a"))
        self.assertTrue(tree.conflict(value, 4, "track-b"))
        self.assertTrue(tree.conflict(value ^ 0b1111, 4, "track-b"))
        self.assertFalse(tree.conflict(value ^ 0b11111, 4, "track-b"))

    def test_color_mapping_is_closed(self):
        self.assertEqual(builder.vehicle_rear_color("grey"), "silver_gray")
        self.assertEqual(builder.vehicle_rear_color("silver"), "silver_gray")
        self.assertEqual(builder.vehicle_rear_color("yellow"), "yellow_orange")
        with self.assertRaises(RuntimeError):
            builder.vehicle_rear_color("purple")

    def test_conservative_body_mapping(self):
        self.assertEqual(builder.vehicle_rear_type("toyota hilux srv"), "light_truck")
        self.assertEqual(builder.vehicle_rear_type("hyundai ix35 2.0"), "suv")
        self.assertEqual(builder.vehicle_rear_type("renault master 2.5"), "van")
        self.assertEqual(builder.vehicle_rear_type("volvo marcopolo viale"), "bus")
        self.assertEqual(builder.vehicle_rear_type("toyota corolla 2.0"), "sedan")
        self.assertIsNone(builder.vehicle_rear_type("fiat palio fire"))

    def test_selected_gzip_tar_payloads_are_read_in_one_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sample.tgz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, payload in (("skip.bin", b"skip"), ("a.png", b"alpha"), ("b.png", b"beta")):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, BytesIO(payload))
            self.assertEqual(
                builder.read_selected_tar_payloads(archive_path, {"a.png", "b.png"}),
                {"a.png": b"alpha", "b.png": b"beta"},
            )


if __name__ == "__main__":
    unittest.main()
