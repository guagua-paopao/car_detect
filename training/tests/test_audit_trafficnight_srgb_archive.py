from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_trafficnight_srgb_archive.py"
SPEC = importlib.util.spec_from_file_location("trafficnight_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (30, 40, 50)).save(buffer, format="JPEG")
    return buffer.getvalue()


def label(label_name: str = "Car") -> bytes:
    return json.dumps({
        "imageWidth": 32,
        "imageHeight": 24,
        "imagePath": "frame.jpg",
        "shapes": [{"label": label_name, "points": [[1, 1], [20, 1], [20, 20], [1, 20]]}],
    }).encode()


class TrafficNightArchiveAuditTest(unittest.TestCase):
    def evidence(self, root: Path) -> Path:
        path = root / "evidence.json"
        path.write_text(json.dumps({"license_spdx": "Apache-2.0"}), encoding="utf-8")
        return path

    def archive(self, root: Path, *, label_name: str = "Car", unsafe: bool = False, omit_label: bool = False) -> Path:
        path = root / "source.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as opened:
            opened.writestr("../escape.jpg" if unsafe else "data/frame.jpg", jpeg_bytes())
            if not omit_label:
                opened.writestr("data/frame.json", label(label_name))
        return path

    def test_passes_paired_official_labelme_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = MODULE.audit_archive(self.archive(root), self.evidence(root), minimum_pairs=1, minimum_objects=1)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["class_counts"], {"Car": 1})

    def test_rejects_unknown_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = MODULE.audit_archive(self.archive(root, label_name="Sedan"), self.evidence(root), minimum_pairs=1, minimum_objects=1)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["unknown_class_counts"], {"Sedan": 1})

    def test_rejects_unsafe_member_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = MODULE.audit_archive(self.archive(root, unsafe=True), self.evidence(root), minimum_pairs=1, minimum_objects=1)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("unsafe ZIP member" in value for value in report["failures"]))

    def test_rejects_unpaired_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = MODULE.audit_archive(self.archive(root, omit_label=True), self.evidence(root), minimum_pairs=1, minimum_objects=1)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["images_without_labels"], 1)


if __name__ == "__main__":
    unittest.main()
