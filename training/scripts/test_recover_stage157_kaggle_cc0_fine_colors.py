import csv
import hashlib
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

from recover_stage157_kaggle_cc0_fine_colors import parse_raw_label, recover_rows


class FineColorRecoveryTests(unittest.TestCase):
    def test_strict_fine_mapping(self):
        self.assertEqual(parse_raw_label("grey_car"), ("gray", "car"))
        self.assertEqual(parse_raw_label("silver_van"), ("silver", "van"))
        self.assertEqual(parse_raw_label("orange_truck"), ("other", "truck"))
        self.assertEqual(parse_raw_label("beige_bus"), ("other", "bus"))

    def test_exact_encoded_crop_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (32, 24), (143, 142, 140))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG", quality=95)
            decoded = Image.open(io.BytesIO(image_bytes.getvalue())).convert("RGB")
            crop_bytes = io.BytesIO()
            decoded.crop((4, 3, 28, 21)).save(crop_bytes, format="JPEG", quality=95, optimize=True)
            crop_sha = hashlib.sha256(crop_bytes.getvalue()).hexdigest()
            annotation = ET.Element("annotation")
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = "silver_car"
            box = ET.SubElement(obj, "bndbox")
            for key, value in (("xmin", 4), ("ymin", 3), ("xmax", 28), ("ymax", 21)):
                ET.SubElement(box, key).text = str(value)
            archive_path = root / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("sample.jpg", image_bytes.getvalue())
                archive.writestr("sample.xml", ET.tostring(annotation))
            rows = [{
                "source_frame_id": "sample.jpg",
                "crop_sha256": crop_sha,
                "color": "silver_gray",
                "color_supervised": "true",
            }]
            recovered, counters = recover_rows(rows, archive_path)
            self.assertEqual(counters["exact_recovered"], 1)
            self.assertEqual(recovered[0]["color"], "silver")
            self.assertEqual(recovered[0]["color_coarse_source"], "silver_gray")
            self.assertEqual(recovered[0]["fine_color_recovery"], "exact_encoded_crop_sha256")


if __name__ == "__main__":
    unittest.main()
