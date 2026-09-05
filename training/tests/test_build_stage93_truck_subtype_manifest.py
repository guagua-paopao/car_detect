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
SCRIPT = ROOT / "scripts" / "build_stage93_truck_subtype_manifest.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stage93_builder", SCRIPT)
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


def image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color).save(path)


def row(path: Path, split: str, body: str, digest: str, dhash: str, **extra: str) -> dict[str, str]:
    output = {
        "image_path": str(path), "split": split, "body_type": body,
        "body_type_supervised": "true", "color": "unknown",
        "color_supervised": "false", "source_dataset": "base",
        "source_label": body, "source_license": "CC-BY-4.0",
        "license_train_eligible": "true", "review_status": "approved",
        "track_group": extra.pop("track_group", ""), "video_id": "",
        "camera_id": "", "sha256": digest, "dhash64": dhash,
        "sample_weight": "1.0", "research_only": "false",
    }
    output.update(extra)
    return output


class Stage93BuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps({
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["light_truck", "heavy_truck", "unknown"],
            "colors": ["unknown"],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def paths_and_args(
        self, base: list[dict[str, str]], stage89: list[dict[str, str]], stage91: list[dict[str, str]]
    ) -> argparse.Namespace:
        base_path = self.root / "stage83.csv"
        p89 = self.root / "stage89.csv"
        p91 = self.root / "stage91.csv"
        write_csv(base_path, base); write_csv(p89, stage89); write_csv(p91, stage91)
        return argparse.Namespace(
            stage83_manifest=base_path,
            expected_stage83_sha256=MODULE.sha256(base_path),
            stage89_manifest=p89,
            expected_stage89_sha256=MODULE.sha256(p89),
            stage91_manifest=p91,
            expected_stage91_sha256=MODULE.sha256(p91),
            labels=self.labels,
            expected_labels_sha256=MODULE.sha256(self.labels),
            safety_root=self.root,
            near_duplicate_hamming=4,
            minimum_train_per_class=1,
            minimum_validation_per_class=1,
            output_manifest=self.root / "out.csv",
            output_report=self.root / "report.json",
        )

    def basic_images(self) -> dict[str, Path]:
        result = {}
        for index, name in enumerate(("lt", "ht", "lv", "hv", "mio", "ina")):
            path = self.root / f"{name}.jpg"
            image(path, (20 + index * 20, 40, 80))
            result[name] = path
        return result

    def test_accepts_only_exact_train_additions_and_never_color(self) -> None:
        p = self.basic_images()
        base = [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
        ]
        stage89 = [row(
            p["mio"], "train", "heavy_truck", "05" * 32, "0f0f0f0f0f0f0f0f",
            source_dataset="MIO-TCD-Classification-2017", source_label="articulated_truck",
            source_license="CC-BY-NC-SA-4.0", review_status="approved_research_only",
        )]
        stage91 = [row(
            p["ina"], "train", "heavy_truck", "06" * 32, "f0f0f0f0f0f0f0f0",
            source_dataset="InaTRC-v1", source_label="three_axle_truck",
            review_status="approved_official_train_bbox_quality_gate",
        )]
        result = MODULE.build(self.paths_and_args(base, stage89, stage91))
        self.assertEqual(result["report"]["status"], "pass")
        self.assertEqual(len(result["rows"]), 6)
        self.assertTrue(all(value["color"] == "unknown" for value in result["rows"]))
        self.assertTrue(all(value["color_supervised"] == "false" for value in result["rows"]))
        mio = next(value for value in result["rows"] if value["stage93_origin"] == "stage89_mio")
        self.assertEqual(mio["deployment_eligible"], "false")

    def test_validation_is_protected_from_near_duplicate_train(self) -> None:
        p = self.basic_images()
        base = [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
        ]
        duplicate = row(
            p["mio"], "train", "heavy_truck", "05" * 32, "00ff00ff00ff00fe",
            source_dataset="MIO-TCD-Classification-2017", source_label="articulated_truck",
            source_license="CC-BY-NC-SA-4.0", review_status="approved_research_only",
        )
        result = MODULE.build(self.paths_and_args(base, [duplicate], []))
        self.assertEqual(result["report"]["status"], "pass")
        self.assertNotIn(str(p["mio"]), {value["image_path"] for value in result["rows"]})
        self.assertIn(str(p["lv"]), {value["image_path"] for value in result["rows"]})
        self.assertEqual(
            result["report"]["integrity"]["dedup"]["counters"]["train_duplicate_or_near_validation"], 1
        )

    def test_group_overlap_filters_train_not_validation(self) -> None:
        p = self.basic_images()
        base = [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff", track_group="same"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
            row(p["mio"], "train", "light_truck", "05" * 32, "0f0f0f0f0f0f0f0f", track_group="same"),
        ]
        result = MODULE.build(self.paths_and_args(base, [], []))
        self.assertEqual(result["report"]["status"], "pass")
        self.assertNotIn(str(p["mio"]), {value["image_path"] for value in result["rows"]})
        self.assertEqual(result["report"]["integrity"]["post_split_leaks"]["group"], 0)

    def test_generic_truck_additions_are_excluded(self) -> None:
        p = self.basic_images()
        base = [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
        ]
        generic = row(
            p["mio"], "train", "truck", "05" * 32, "0f0f0f0f0f0f0f0f",
            source_dataset="MIO-TCD-Classification-2017", source_label="single_unit_truck",
            source_license="CC-BY-NC-SA-4.0", review_status="approved_research_only",
        )
        result = MODULE.build(self.paths_and_args(base, [generic], []))
        self.assertEqual(result["report"]["status"], "pass")
        self.assertTrue(all(value["body_type"] in MODULE.FINE_LABELS for value in result["rows"]))
        self.assertEqual(result["report"]["selection"]["generic_truck_rows_allowed"], 0)

    def test_hash_mismatch_fails_before_reading_rows(self) -> None:
        p = self.basic_images()
        base = [
            row(p["lt"], "train", "light_truck", "01" * 32, "0000000000000000"),
            row(p["ht"], "train", "heavy_truck", "02" * 32, "ffffffffffffffff"),
            row(p["lv"], "validation", "light_truck", "03" * 32, "00ff00ff00ff00ff"),
            row(p["hv"], "validation", "heavy_truck", "04" * 32, "ff00ff00ff00ff00"),
        ]
        args = self.paths_and_args(base, [], [])
        args.expected_stage83_sha256 = "0" * 64
        with self.assertRaises(RuntimeError):
            MODULE.build(args)

    def test_validation_internal_conflict_removes_both_rows(self) -> None:
        p = self.basic_images()
        rows = [
            {**row(p["lv"], "validation", "light_truck", "01" * 32, "00ff00ff00ff00ff"), "stage93_origin": "stage83"},
            {**row(p["hv"], "validation", "heavy_truck", "02" * 32, "00ff00ff00ff00fe"), "stage93_origin": "stage83"},
        ]
        output, audit = MODULE.deduplicate_validation_first(rows, 4)
        self.assertEqual(output, [])
        self.assertEqual(audit["counters"]["validation_conflict_rows_removed"], 2)
        self.assertEqual(audit["validation_rows_after_dedup"], 0)


if __name__ == "__main__":
    unittest.main()
