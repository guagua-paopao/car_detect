from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TRAINING_ROOT / "scripts"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def body_row(path: str, label: str) -> dict[str, str]:
    return {
        "image_path": path,
        "split": "train",
        "formal_train_eligible": "true",
        "teacher_consensus": "accepted",
        "body_type_supervised": "true",
        "body_type": label,
        "color": "unknown",
    }


def body_report(manifest: Path) -> dict:
    return {
        "status": "pass",
        "output_manifest_sha256": file_sha256(manifest),
        "accepted_rows": 1,
        "policy": {
            "all_teachers_exact_agreement": True,
            "all_views_exact_agreement": True,
            "official_coarse_class_compatibility_required": True,
            "frozen_video_used": False,
        },
    }


class NightPhotometricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = SCRIPTS / "build_openimages_machine_night_crops.py"
        spec = importlib.util.spec_from_file_location("night_crop_builder", path)
        assert spec and spec.loader
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_dark_image_scores_higher_than_bright_image(self) -> None:
        dark = self.module.photometrics(Image.new("RGB", (128, 96), (20, 20, 20)))
        bright = self.module.photometrics(Image.new("RGB", (128, 96), (220, 220, 220)))
        self.assertGreater(dark["lowlight_score"], bright["lowlight_score"])
        self.assertGreater(dark["dark_fraction"], bright["dark_fraction"])
        self.assertLess(dark["mean"], bright["mean"])


class ForegroundLightingPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = SCRIPTS / "build_openimages_foreground_color_proposals.py"
        spec = importlib.util.spec_from_file_location("foreground_proposals", path)
        assert spec and spec.loader
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_verified_machine_night_lighting_is_preserved(self) -> None:
        value = self.module.preserve_verified_lighting(
            "night_machine_photometric_consensus", False
        )
        self.assertEqual(value, "night_machine_photometric_consensus")

    def test_proxy_is_used_only_when_upstream_lighting_is_unknown(self) -> None:
        self.assertEqual(
            self.module.preserve_verified_lighting("unknown", True), "low_light_proxy"
        )

    def test_per_image_cc_by_attribution_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attribution.csv"
            write_csv(path, [{
                "image_id": "abc",
                "license_url": "https://creativecommons.org/licenses/by/2.0/",
            }])
            self.assertEqual(self.module.validate_attribution(path, {"abc"}), 1)
            with self.assertRaises(RuntimeError):
                self.module.validate_attribution(path, {"abc", "missing"})


class TeacherManifestCombinerTests(unittest.TestCase):
    def run_combiner(self, root: Path, second_label: str = "suv") -> subprocess.CompletedProcess:
        first_manifest = root / "first.csv"
        second_manifest = root / "second.csv"
        write_csv(first_manifest, [body_row("first.jpg", "sedan")])
        write_csv(second_manifest, [body_row("second.jpg", second_label)])
        first_report = root / "first.json"
        second_report = root / "second.json"
        first_report.write_text(json.dumps(body_report(first_manifest)), encoding="utf-8")
        second_report.write_text(json.dumps(body_report(second_manifest)), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "combine_attribute_teacher_consensus_manifests.py"),
                "--head", "body",
                "--manifest", str(first_manifest),
                "--report", str(first_report),
                "--pool", "weather",
                "--manifest", str(second_manifest),
                "--report", str(second_report),
                "--pool", "machine_night",
                "--output-manifest", str(root / "combined.csv"),
                "--output-report", str(root / "combined.json"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_combines_only_preaccepted_hash_verified_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_combiner(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "combined.json").read_text(encoding="utf-8"))
            self.assertEqual(report["accepted_rows"], 2)
            self.assertEqual(report["accepted_pool_counts"], {"machine_night": 1, "weather": 1})
            self.assertTrue(report["policy"]["only_previously_accepted_rows_retained"])

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_manifest = root / "first.csv"
            write_csv(first_manifest, [body_row("first.jpg", "sedan")])
            report_path = root / "first.json"
            report = body_report(first_manifest)
            report["output_manifest_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "combine_attribute_teacher_consensus_manifests.py"),
                    "--head", "body",
                    "--manifest", str(first_manifest),
                    "--report", str(report_path),
                    "--pool", "weather",
                    "--output-manifest", str(root / "combined.csv"),
                    "--output-report", str(root / "combined.json"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "combined.csv").exists())


if __name__ == "__main__":
    unittest.main()
