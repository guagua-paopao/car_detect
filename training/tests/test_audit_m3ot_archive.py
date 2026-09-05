from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_m3ot_archive.py"
SPEC = importlib.util.spec_from_file_location("m3ot_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def coco(drone: str, *, bad_box: bool = False) -> bytes:
    image = {
        "id": 1,
        "video_id": 1,
        "file_name": f"/dataset/train/{drone}/{drone}-01/img1/000001.PNG",
        "height": 32,
        "width": 48,
        "frame_id": 0,
        "mot_frame_id": 1,
    }
    annotation = {
        "category_id": 1,
        "bbox": [45, 1, 10, 10] if bad_box else [1, 1, 10, 10],
        "area": 100,
        "iscrowd": False,
        "visibility": 1.0,
        "mot_instance_id": 1,
        "mot_conf": 1.0,
        "mot_class_id": 1,
        "id": 1,
        "image_id": 1,
        "instance_id": 1,
    }
    return json.dumps({"images": [image], "annotations": [annotation], "categories": [{"id": 1, "name": "vehicle"}], "videos": []}).encode()


class M3OTArchiveAuditTests(unittest.TestCase):
    def make(self, root: Path, *, unsafe: bool = False, bad_box: bool = False) -> tuple[Path, Path]:
        archive = root / "M3OT.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as opened:
            for drone in ("1", "2"):
                opened.writestr(f"M3OT/Annotations/{drone}/rgb/train_cocoformat.json", coco(drone, bad_box=bad_box))
                opened.writestr(f"M3OT/{drone}/rgb/train/{drone}-01/img1/000001.PNG", b"png-not-opened-by-audit")
            if unsafe:
                opened.writestr("../escape.txt", b"bad")
        md5, _ = MODULE.digest_file(archive)
        evidence = root / "evidence.json"
        evidence.write_text(json.dumps({"license": {"name": "CC BY 4.0"}, "archive": {"bytes": archive.stat().st_size, "supplied_md5": md5}}), encoding="utf-8")
        return archive, evidence

    def audit(self, archive: Path, evidence: Path) -> dict[str, object]:
        md5, _ = MODULE.digest_file(archive)
        return MODULE.audit_archive(archive, evidence, expected_bytes=archive.stat().st_size, expected_md5=md5, minimum_rgb_train_images=2, minimum_rgb_train_annotations=2)

    def test_pass_reads_train_annotations_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, evidence = self.make(Path(directory))
            report = self.audit(archive, evidence)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["rgb_train_coco_images"], 2)
            self.assertEqual(report["rgb_train_coco_annotations"], 2)
            self.assertFalse(report["validation_payload_opened"])
            self.assertFalse(report["test_payload_opened"])

    def test_rejects_unsafe_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, evidence = self.make(Path(directory), unsafe=True)
            report = self.audit(archive, evidence)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["unsafe_member_paths"], 1)

    def test_rejects_out_of_bounds_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, evidence = self.make(Path(directory), bad_box=True)
            report = self.audit(archive, evidence)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["rgb_train_coco_annotations"], 0)


if __name__ == "__main__":
    unittest.main()
