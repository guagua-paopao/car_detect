from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage71_teacher_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_stage71_teacher_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path", "body_type", "color", "body_type_supervised", "color_supervised",
    "coarse_body_family", "split", "source_dataset", "track_group", "video_id",
    "source_group", "crop_sha256", "crop_dhash64", "sample_weight",
    "night", "low_light", "lighting",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(image: Path, *, split: str, color: str, source: str, group: str, phash: str) -> dict[str, str]:
    return {
        "image_path": str(image),
        "body_type": "sedan",
        "color": color,
        "body_type_supervised": "true",
        "color_supervised": "true",
        "coarse_body_family": "",
        "split": split,
        "source_dataset": source,
        "track_group": group,
        "video_id": group,
        "source_group": group,
        "crop_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "crop_dhash64": phash,
        "sample_weight": "1.0",
        "night": "false",
        "low_light": "false",
        "lighting": "daylight",
    }


class Stage71ManifestTests(unittest.TestCase):
    def run_builder(self, root: Path, dvm_rows: list[dict[str, str]], cctv_rows: list[dict[str, str]]) -> tuple[int, dict]:
        body_manifest = root / "body.csv"
        dvm_manifest = root / "dvm.csv"
        cctv_manifest = root / "cctv.csv"
        write_csv(body_manifest, [dict(cctv_rows[0], color="unknown", color_supervised="false")])
        write_csv(dvm_manifest, dvm_rows)
        write_csv(cctv_manifest, cctv_rows)
        output = root / "out"
        argv = [
            str(SCRIPT),
            "--body-manifest", str(body_manifest), "--body-root", str(root),
            "--dvm-manifest", str(dvm_manifest), "--dvm-root", str(root),
            "--cctv-color-manifest", str(cctv_manifest), "--cctv-color-root", str(root),
            "--datasets-safety-root", str(root), "--output-root", str(output),
            "--expected-body-sha256", digest(body_manifest),
            "--expected-dvm-sha256", digest(dvm_manifest),
            "--expected-cctv-color-sha256", digest(cctv_manifest),
            "--minimum-color-train-rows", "2",
            "--cctv-sample-weight", "10",
        ]
        with mock.patch("sys.argv", argv):
            code = MODULE.main()
        report = json.loads((output / "stage71-teacher-manifests-report.json").read_text(encoding="utf-8"))
        return code, report

    def test_builds_replay_manifest_and_prefers_cctv_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            for index in range(4):
                image = root / f"{index}.jpg"
                image.write_bytes(f"image-{index}".encode())
                images.append(image)
            dvm = [
                row(images[0], split="train", color="white", source="DVM", group="dvm-0", phash="0000000000000000"),
                row(images[1], split="train", color="black", source="DVM", group="dvm-1", phash="ffffffffffffffff"),
            ]
            cctv = [
                dict(row(images[2], split="train", color="red", source="CCTV", group="cctv-train", phash="0f0f0f0f0f0f0f0f"), lighting="low_light_proxy"),
                row(images[3], split="validation", color="white", source="CCTV", group="cctv-val", phash="f0f0f0f0f0f0f0f0"),
            ]
            code, report = self.run_builder(root, dvm, cctv)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["outputs"]["color_teacher"]["splits"], {"train": 3, "validation": 1})
            self.assertGreater(report["outputs"]["color_teacher"]["expected_weighted_cctv_fraction_before_class_weighting"], 0.8)
            manifest = Path(report["outputs"]["color_teacher"]["path"])
            with manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(item["body_type"] == "unknown" for item in rows))
            self.assertTrue(all(item["body_type_supervised"] == "false" for item in rows))
            self.assertEqual(next(item for item in rows if item["stage71_origin"] == "cctv" and item["split"] == "train")["night"], "true")

    def test_rejects_frozen_asset_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "vcas_rtsp_demo_60s.jpg"
            image.write_bytes(b"frozen")
            dvm = [row(image, split="train", color="white", source="DVM", group="dvm", phash="0000000000000000")]
            cctv = [row(image, split="validation", color="white", source="CCTV", group="val", phash="ffffffffffffffff")]
            with self.assertRaisesRegex(RuntimeError, "frozen asset marker"):
                self.run_builder(root, dvm, cctv)

    def test_conflicting_exact_duplicate_removes_both(self) -> None:
        left = {"image_path": "left", "split": "validation", "stage71_origin": "cctv", "color": "white", "crop_sha256": "same", "crop_dhash64": "0000000000000000"}
        right = {"image_path": "right", "split": "train", "stage71_origin": "dvm", "color": "black", "crop_sha256": "same", "crop_dhash64": "0000000000000001"}
        output, report = MODULE.deduplicate_color([left, right], 3)
        self.assertEqual(output, [])
        self.assertEqual(report["color_conflicts_removed_both"], {"white->black": 1})

    def test_cross_color_grayscale_near_match_is_not_a_duplicate(self) -> None:
        left = {"image_path": "left", "split": "validation", "stage71_origin": "cctv", "color": "white", "crop_sha256": "a", "crop_dhash64": "0000000000000000"}
        right = {"image_path": "right", "split": "train", "stage71_origin": "dvm", "color": "black", "crop_sha256": "b", "crop_dhash64": "0000000000000001"}
        output, report = MODULE.deduplicate_color([left, right], 3)
        self.assertEqual(len(output), 2)
        self.assertEqual(report["color_conflicts_removed_both"], {})


if __name__ == "__main__":
    unittest.main()
