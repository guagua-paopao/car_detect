from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_stage172_trafficnight_train_crops import (  # noqa: E402
    SOURCE_MAPPING,
    dhash64,
    encode_jpeg,
    pair_members,
    polygon_crop,
    scale_declared_points,
    validate_polygon,
)


class Stage172TrafficNightCropTests(unittest.TestCase):
    def test_mapping_does_not_fabricate_car_or_trailer_fine_types(self) -> None:
        self.assertEqual(SOURCE_MAPPING["Car"]["body_type"], "unknown")
        self.assertEqual(SOURCE_MAPPING["Car"]["coarse_body_family"], "car")
        self.assertFalse(SOURCE_MAPPING["Car"]["fine_supervised"])
        self.assertEqual(SOURCE_MAPPING["Tractor"]["body_type"], "unknown")
        self.assertEqual(SOURCE_MAPPING["Tractor"]["coarse_body_family"], "truck")
        self.assertEqual(SOURCE_MAPPING["Buses"]["coarse_body_family"], "")
        self.assertEqual(SOURCE_MAPPING["Semi-Trailers"]["coarse_body_family"], "")
        self.assertEqual(SOURCE_MAPPING["Empty Semi-Trailers"]["coarse_body_family"], "")

    def test_validate_polygon_rejects_out_of_bounds_instead_of_clipping(self) -> None:
        with self.assertRaisesRegex(ValueError, "out_of_bounds_polygon"):
            validate_polygon([[0, 0], [10, 0], [10, 10], [-1, 10]], 20, 20)

    def test_polygon_crop_clamps_only_padding(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        points = validate_polygon([[0, 0], [20, 0], [20, 10], [0, 10]], 100, 80)
        crop = polygon_crop(image, points, 0.10)
        self.assertEqual(crop.size, (23, 12))

    def test_declared_dimension_transform_matches_decoded_image(self) -> None:
        self.assertEqual(
            scale_declared_points([[0, 0], [4000, 3000], [2000, 1500]], 4000, 3000, 2776, 2208),
            [[0, 0], [2776, 2208], [1388, 1104]],
        )

    def test_manifest_dhash_contract_is_based_on_persisted_jpeg(self) -> None:
        import io

        source = Image.new("RGB", (31, 23))
        for y in range(source.height):
            for x in range(source.width):
                source.putpixel((x, y), ((x * 19 + y * 7) % 256, (x * 3 + y * 29) % 256, (x * 13 + y * 11) % 256))
        encoded = encode_jpeg(source)
        with Image.open(io.BytesIO(encoded)) as persisted:
            persisted.load()
            expected = dhash64(persisted)
        self.assertEqual(expected, dhash64(Image.open(io.BytesIO(encoded))))

    def test_pair_members_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
            pair_members(["../escape.jpg"])

    def test_pair_members_pairs_by_full_stem(self) -> None:
        images, annotations = pair_members([
            "root/a.JPG", "root/a.json", "root/b.png", "root/b.json",
        ])
        self.assertEqual(set(images), {"root/a", "root/b"})
        self.assertEqual(set(annotations), {"root/a", "root/b"})


if __name__ == "__main__":
    unittest.main()
