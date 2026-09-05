from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_stage98_two_axle_cctv_specialist_manifest.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stage98_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "color",
    "color_supervised", "source_dataset", "source_label", "source_license",
    "license_train_eligible", "review_status", "track_group", "video_id",
    "camera_id", "sha256", "dhash64", "sample_weight", "research_only",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color).save(path)


def row(path: Path, split: str, body: str, digest: str, dhash: str, **extra: str) -> dict[str, str]:
    output = {
        "image_path": str(path), "split": split, "body_type": body,
        "body_type_supervised": "true", "color": "unknown",
        "color_supervised": "false", "source_dataset": "base",
        "source_label": body, "source_license": "CC BY 4.0",
        "license_train_eligible": "true", "review_status": "approved",
        "track_group": extra.pop("track_group", ""), "video_id": "",
        "camera_id": "", "sha256": digest, "dhash64": dhash,
        "sample_weight": "1.0", "research_only": "false",
    }
    output.update(extra)
    return output


class Stage98BuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps({
            "body_types": ["light_truck", "heavy_truck", "unknown"],
            "colors": ["unknown"],
        }), encoding="utf-8")
        self.evidence = self.root / "evidence.json"
        self.evidence.write_text(json.dumps({
            "license": "CC BY 4.0",
            "class_contract": {"1": "Class-2 truck with two axles"},
            "policy": {"original_validation_and_test_payloads_train_eligible": False},
        }), encoding="utf-8")
        self.report = self.root / "stage91-report.json"
        self.report.write_text(json.dumps({
            "status": "pass",
            "policy": {"source_validation_payloads_read": 0, "source_test_payloads_read": 0},
        }), encoding="utf-8")
        self.images: dict[str, Path] = {}
        for index, name in enumerate(("lt", "ht", "lv", "hv", "two", "three")):
            path = self.root / f"{name}.jpg"
            make_image(path, (20 + index * 20, 40, 80))
            self.images[name] = path

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, stage93: list[dict[str, str]], stage91: list[dict[str, str]]) -> argparse.Namespace:
        p93 = self.root / "stage93.csv"
        p91 = self.root / "stage91.csv"
        write_csv(p93, stage93)
        write_csv(p91, stage91)
        return argparse.Namespace(
            stage93_manifest=p93,
            expected_stage93_sha256=MODULE.sha256(p93),
            stage91_manifest=p91,
            expected_stage91_sha256=MODULE.sha256(p91),
            stage91_report=self.report,
            expected_stage91_report_sha256=MODULE.sha256(self.report),
            source_evidence=self.evidence,
            expected_source_evidence_sha256=MODULE.sha256(self.evidence),
            labels=self.labels,
            expected_labels_sha256=MODULE.sha256(self.labels),
            safety_root=self.root,
            near_duplicate_hamming=4,
            minimum_train_per_class=1,
            minimum_validation_per_class=1,
            minimum_two_axle_rows=1,
            output_manifest=self.root / "out.csv",
            output_report=self.root / "out.json",
        )

    def base(self) -> list[dict[str, str]]:
        p = self.images
        return [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
        ]

    def test_maps_only_official_two_axle_train_rows_to_light_truck(self) -> None:
        p = self.images
        stage91 = [
            row(p["two"], "train", "truck", "05" * 32, "0f0f0f0f0f0f0f0f", source_label="two_axle_truck"),
            row(p["three"], "train", "heavy_truck", "06" * 32, "f0f0f0f0f0f0f0f0", source_label="three_axle_truck"),
        ]
        result = MODULE.build(self.args(self.base(), stage91))
        self.assertEqual(result["report"]["status"], "pass")
        added = [value for value in result["rows"] if value.get("stage98_origin")]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["body_type"], "light_truck")
        self.assertEqual(added[0]["stage98_fine_truth_source"], "official_class_2_two_axle_toll_cctv")

    def test_validation_near_duplicate_has_priority_over_two_axle_addition(self) -> None:
        p = self.images
        addition = row(
            p["two"], "train", "truck", "05" * 32, "00ff00ff00ff00fe",
            source_label="two_axle_truck",
        )
        result = MODULE.build(self.args(self.base(), [addition]))
        self.assertEqual(result["report"]["status"], "pass")
        self.assertNotIn(str(p["two"]), {value["image_path"] for value in result["rows"]})
        self.assertEqual(result["report"]["integrity"]["post_split_leaks"]["near"], 0)

    def test_source_evidence_must_explicitly_say_two_axles(self) -> None:
        self.evidence.write_text(json.dumps({
            "license": "CC BY 4.0", "class_contract": {"1": "truck"}, "policy": {},
        }), encoding="utf-8")
        args = self.args(self.base(), [])
        args.expected_source_evidence_sha256 = MODULE.sha256(self.evidence)
        with self.assertRaises(RuntimeError):
            MODULE.build(args)

    def test_hash_mismatch_fails_closed(self) -> None:
        args = self.args(self.base(), [])
        args.expected_stage93_sha256 = "0" * 64
        with self.assertRaises(RuntimeError):
            MODULE.build(args)


if __name__ == "__main__":
    unittest.main()
