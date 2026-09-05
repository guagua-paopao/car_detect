from __future__ import annotations

import argparse
import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_stage95_color_consistency_manifest.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stage95_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FIELDS = [
    "image_path", "split", "body_type", "color", "body_type_supervised",
    "color_supervised", "license_train_eligible", "review_status", "night",
    "lighting", "vehicle_size", "occluded", "truncated", "blur", "crop_sha256",
    "dhash64", "source_license", "pseudo_label", "pseudo_label_confidence",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def make_image(path: Path, value: int) -> None:
    Image.new("RGB", (20, 20), (value, value, value)).save(path)


def source_row(path: str, digest: str, dhash: str, **extra: str) -> dict[str, str]:
    row = {
        "image_path": path, "split": "train", "body_type": "sedan", "color": "red",
        "body_type_supervised": "true", "color_supervised": "true",
        "license_train_eligible": "true", "review_status": "pending",
        "night": "true", "lighting": "night_machine_photometric_consensus",
        "vehicle_size": "small", "occluded": "false", "truncated": "false",
        "blur": "false", "crop_sha256": digest, "dhash64": dhash,
        "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
        "pseudo_label": "true", "pseudo_label_confidence": "0.99",
    }
    row.update(extra)
    return row


class Stage95BuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, base: list[dict[str, str]], source: list[dict[str, str]]) -> argparse.Namespace:
        base_path = self.root / "base.csv"; source_path = self.root / "source.csv"
        write_csv(base_path, base); write_csv(source_path, source)
        return argparse.Namespace(
            stage78_manifest=base_path, expected_stage78_sha256=MODULE.sha256(base_path),
            stage61_manifest=source_path, expected_stage61_sha256=MODULE.sha256(source_path),
            datasets_root=self.root, near_duplicate_hamming=4,
            minimum_rows=1, minimum_night_rows=1,
            output_manifest=self.root / "out.csv", output_report=self.root / "report.json",
        )

    def test_resets_all_attribute_truth_to_unknown(self) -> None:
        image = self.root / "night.jpg"; make_image(image, 30)
        result = MODULE.build(self.args([], [source_row("night.jpg", "01" * 32, "00ff00ff00ff00ff")]))
        self.assertEqual(result["report"]["status"], "pass")
        row = result["rows"][0]
        self.assertEqual((row["body_type"], row["color"]), ("unknown", "unknown"))
        self.assertEqual((row["body_type_supervised"], row["color_supervised"]), ("false", "false"))
        self.assertEqual(row["pseudo_label"], "false")

    def test_filters_near_overlap_with_color_training_manifest(self) -> None:
        image = self.root / "night.jpg"; make_image(image, 30)
        base = [source_row("night.jpg", "02" * 32, "00ff00ff00ff00fe")]
        source = [source_row("night.jpg", "01" * 32, "00ff00ff00ff00ff")]
        args = self.args(base, source); args.minimum_rows = 0; args.minimum_night_rows = 0
        result = MODULE.build(args)
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["report"]["output"]["counters"]["near_overlap_stage78"], 1)

    def test_deduplicates_candidate_pool(self) -> None:
        first = self.root / "a.jpg"; second = self.root / "b.jpg"
        make_image(first, 20); make_image(second, 21)
        source = [
            source_row("a.jpg", "01" * 32, "00ff00ff00ff00ff"),
            source_row("b.jpg", "02" * 32, "00ff00ff00ff00fe"),
        ]
        result = MODULE.build(self.args([], source))
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["report"]["output"]["counters"]["candidate_near_duplicate"], 1)

    def test_unverified_lighting_does_not_count_as_night(self) -> None:
        image = self.root / "dark.jpg"; make_image(image, 10)
        source = [source_row("dark.jpg", "01" * 32, "00ff00ff00ff00ff", lighting="low_light_proxy")]
        args = self.args([], source); args.minimum_night_rows = 1
        result = MODULE.build(args)
        self.assertEqual(result["report"]["status"], "fail")
        self.assertEqual(result["report"]["output"]["verified_night_rows"], 0)

    def test_hash_mismatch_fails_closed(self) -> None:
        image = self.root / "night.jpg"; make_image(image, 30)
        args = self.args([], [source_row("night.jpg", "01" * 32, "00ff00ff00ff00ff")])
        args.expected_stage61_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Stage61 immutable"):
            MODULE.build(args)


if __name__ == "__main__":
    unittest.main()
