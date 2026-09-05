from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "training" / "scripts" / "build_stage72_body_taxonomy_v2_manifest.py"
LABELS = ROOT / "config" / "vehicle_labels.v2.json"
EXACT = ["sedan", "suv", "mpv", "van", "pickup", "bus", "light_truck", "heavy_truck", "other"]
FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "source_dataset", "source_group", "crop_sha256", "crop_dhash64",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_row(image: Path, split: str, label: str, index: int, *, coarse: str = "") -> dict[str, str]:
    return {
        "image_path": str(image),
        "split": split,
        "body_type": label,
        "body_type_supervised": "true" if label != "unknown" else "false",
        "coarse_body_family": coarse,
        "source_dataset": "licensed-real-cctv",
        "source_group": f"{split}-group-{index}",
        "crop_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "crop_dhash64": f"{index + (0 if split == 'train' else 1000):016x}",
    }


class Stage72BodyManifestTest(unittest.TestCase):
    def run_builder(self, root: Path, rows: list[dict[str, str]]) -> subprocess.CompletedProcess[str]:
        source = root / "stage71-body.csv"
        write_manifest(source, rows)
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--input-manifest", str(source),
                "--expected-input-sha256", digest(source),
                "--labels", str(LABELS),
                "--expected-labels-sha256", digest(LABELS),
                "--datasets-safety-root", str(root),
                "--output-manifest", str(root / "stage72-body.csv"),
                "--output-report", str(root / "report.json"),
                "--near-duplicate-hamming", "0",
                "--minimum-train-rows", "10",
                "--minimum-validation-rows", "9",
            ],
            text=True, capture_output=True, check=False,
        )

    def valid_rows(self, root: Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        index = 0
        for split in ("train", "validation"):
            for label in EXACT:
                image = root / f"{split}-{label}.jpg"
                image.write_bytes(f"{split}-{label}".encode())
                rows.append(make_row(image, split, label, index))
                index += 1
        image = root / "coarse-truck.jpg"
        image.write_bytes(b"coarse-truck")
        rows.append(make_row(image, "train", "truck", index, coarse="truck"))
        return rows

    def test_preserves_exact_truth_and_converts_generic_truck_to_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_builder(root, self.valid_rows(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "stage72-body.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            truck = next(row for row in rows if row["coarse_body_family"] == "truck")
            self.assertEqual(truck["body_type"], "unknown")
            self.assertEqual(truck["body_type_supervised"], "false")
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["policy"]["generic_truck_is_conservative_decode_fallback"])
            self.assertFalse(report["policy"]["frozen_video_used"])

    def test_cross_split_group_leak_fails_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.valid_rows(root)
            rows[9]["source_group"] = rows[0]["source_group"]
            result = self.run_builder(root, rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "stage72-body.csv").exists())

    def test_exact_image_leak_fails_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.valid_rows(root)
            rows[9]["crop_sha256"] = rows[0]["crop_sha256"]
            result = self.run_builder(root, rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "stage72-body.csv").exists())

    def test_frozen_or_test_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.valid_rows(root)
            rows[0]["image_path"] = str(root / "vcas_rtsp_demo_60s.jpg")
            result = self.run_builder(root, rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "stage72-body.csv").exists())


if __name__ == "__main__":
    unittest.main()
