from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage70_uadetrac_training_pool.py"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sequence(root: Path, xml_root: Path, sequence: str, vehicle_type: str, color: str = "unknown") -> None:
    image_dir = root / f"detrac_1_{sequence}"
    image_dir.mkdir(parents=True)
    frames = []
    fill = (20, 60, 180) if vehicle_type == "bus" else (180, 70, 20)
    for number in range(1, 8):
        image = np.zeros((140, 240, 3), dtype=np.uint8)
        cv2.rectangle(image, (20 + number * 3, 30), (130 + number * 3, 110), fill, -1)
        if vehicle_type == "car":
            cv2.circle(image, (70 + number * 3, 70), 14, (240, 240, 240), -1)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        encoded.tofile(str(image_dir / f"img{number:05d}.jpg"))
        color_attribute = "" if color == "unknown" else f' color="{color}"'
        frames.append(
            f'<frame num="{number}"><target_list><target id="1">'
            f'<box left="{20 + number * 3}" top="30" width="110" height="80"/>'
            f'<attribute truncation_ratio="0" vehicle_type="{vehicle_type}"{color_attribute}/>'
            f'</target></target_list></frame>'
        )
    xml_root.mkdir(parents=True, exist_ok=True)
    (xml_root / f"{sequence}.xml").write_text(
        f'<sequence name="{sequence}"><sequence_attribute camera_state="stable" weather="night"/>'
        + "".join(frames) + "</sequence>", encoding="utf-8"
    )


class Stage70UADetracTrainingPoolTest(unittest.TestCase):
    def test_training_only_exact_coarse_and_color_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            training = tmp / "training"
            xml_root = tmp / "training_xml"
            training.mkdir()
            make_sequence(training, xml_root, "MVI_20011", "bus", color="red")
            make_sequence(training, xml_root, "MVI_20012", "car")
            archive = tmp / "ua_detrac_training_set.zip"
            archive.write_bytes(b"immutable-training-archive-fixture")
            source_index = tmp / "source-index.json"
            source_index.write_text('{"partition":"training"}\n', encoding="utf-8")
            license_evidence = tmp / "README.md"
            license_evidence.write_text("mirror-declared CC BY 4.0\n", encoding="utf-8")
            output = tmp / "output"
            completed = subprocess.run([
                sys.executable, str(SCRIPT),
                "--training-root", str(training), "--training-xml-root", str(xml_root),
                "--source-archive", str(archive),
                "--expected-source-archive-sha256", file_sha(archive),
                "--source-index", str(source_index), "--license-evidence", str(license_evidence),
                "--output-root", str(output), "--window", "3", "--windows-per-track", "1",
                "--near-duplicate-hamming", "0", "--workers", "2",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
            with (output / "attribute_manifest.training-only.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["policy"]["test_arguments_supported"])
            self.assertFalse(report["policy"]["test_accessed"])
            self.assertEqual(report["color_supervised_rows"], 0)
            self.assertTrue(all(row["color"] == "unknown" for row in rows))
            self.assertTrue(all(row["color_supervised"] == "false" for row in rows))
            bus = [row for row in rows if row["source_vehicle_type"] == "bus"]
            car = [row for row in rows if row["source_vehicle_type"] == "car"]
            self.assertTrue(bus and car)
            self.assertTrue(all(row["body_type"] == "bus" and row["body_type_supervised"] == "true" for row in bus))
            self.assertTrue(all(row["body_type"] == "unknown" and row["coarse_body_family"] == "car" for row in car))
            self.assertEqual(report["video_leaks"], [])
            self.assertEqual(report["track_leaks"], [])
            self.assertEqual(report["deduplication"]["post_filter_cross_split_count"], 0)

    def test_post_filter_cross_split_near_duplicate_audit(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage70_pool", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [
            {"split": "train", "image_path": "train/a.jpg", "dhash64": "0000000000000000"},
            {"split": "validation", "image_path": "validation/b.jpg", "dhash64": "0000000000000003"},
        ]
        examples = module.cross_split_near_duplicate_examples(rows, threshold=2)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["distance"], 2)

    def test_rejects_test_path_and_bad_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            test_root = tmp / "test"
            xml_root = tmp / "training_xml"
            test_root.mkdir(); xml_root.mkdir()
            archive = tmp / "archive.zip"; archive.write_bytes(b"x")
            source_index = tmp / "source-index.json"; source_index.write_text("{}")
            evidence = tmp / "README.md"; evidence.write_text("license")
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--training-root", str(test_root),
                "--training-xml-root", str(xml_root), "--source-archive", str(archive),
                "--expected-source-archive-sha256", "0" * 64, "--source-index", str(source_index),
                "--license-evidence", str(evidence), "--output-root", str(tmp / "output"),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not a training-only path", completed.stderr)


if __name__ == "__main__":
    unittest.main()
