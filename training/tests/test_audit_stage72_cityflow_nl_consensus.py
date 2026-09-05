from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "audit_stage72_cityflow_nl_consensus.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(descriptions: list[str], prefix: str) -> dict:
    return {
        "nl": descriptions,
        "nl_other_views": [],
        "frames": [f"train/S01/c001/{prefix}-1.jpg", f"train/S01/c001/{prefix}-2.jpg"],
        "boxes": [[0, 0, 20, 20], [1, 1, 20, 20]],
    }


class Stage72CityFlowNlConsensusAuditTest(unittest.TestCase):
    def run_audit(self, root: Path, tracks: dict, expected_tracks: int | None = None) -> subprocess.CompletedProcess[str]:
        annotations = root / "train-tracks.json"
        license_file = root / "LICENSE"
        annotations.write_text(json.dumps(tracks), encoding="utf-8")
        license_file.write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--annotations",
                str(annotations),
                "--expected-annotations-sha256",
                sha256(annotations),
                "--license",
                str(license_file),
                "--expected-license-sha256",
                sha256(license_file),
                "--repository-commit",
                "test-commit",
                "--output-labels",
                str(root / "labels.jsonl"),
                "--output-report",
                str(root / "report.json"),
                "--expected-track-count",
                str(expected_tracks if expected_tracks is not None else len(tracks)),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_two_agreeing_single_label_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_audit(
                root,
                {
                    "t1": track(
                        ["A gray sedan turns left.", "The gray sedan continues.", "A car is moving."],
                        "a",
                    )
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            label = json.loads((root / "labels.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(label["color"], "gray")
            self.assertEqual(label["body_type"], "sedan")
            self.assertEqual(label["color_consensus_descriptions"], 2)

    def test_rejects_conflicting_and_multi_vehicle_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_audit(
                root,
                {
                    "t1": track(
                        [
                            "A gray sedan follows a silver truck.",
                            "A gray sedan drives forward.",
                            "A silver sedan drives forward.",
                        ],
                        "b",
                    )
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            label = json.loads((root / "labels.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(label["color"], "unknown")
            self.assertEqual(label["body_type"], "sedan")

    def test_non_target_paint_names_remain_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_audit(
                root,
                {
                    "t1": track(
                        ["An orange sedan turns left.", "The orange sedan continues.", "Orange car."],
                        "orange",
                    )
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            label = json.loads((root / "labels.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(label["color"], "unknown")

    def test_rejects_duplicate_frame_and_box_observations_across_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = track(["A red SUV.", "The red SUV.", "Red vehicle."], "same")
            result = self.run_audit(root, {"t1": duplicate, "t2": duplicate})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "labels.jsonl").exists())

    def test_fails_on_frozen_marker_or_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_audit(
                root,
                {
                    "t1": track(
                        ["vcas_rtsp_demo_60s gray sedan", "gray sedan", "gray sedan"],
                        "frozen",
                    )
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "labels.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
