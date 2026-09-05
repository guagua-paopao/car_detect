from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "audit_stage88_unlabeled_hard_scene_reservoir.py"
SPEC = importlib.util.spec_from_file_location("stage88_reservoir", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path", "body_type", "body_type_supervised", "color", "color_supervised",
    "split", "review_status", "crop_sha256", "night", "low_light", "lighting",
    "small_target", "vehicle_size", "occluded", "truncated", "occlusion_level",
    "hard_mining_tags", "license_train_eligible", "source_dataset", "source_license",
]


def row(index: int, *, supervised: bool) -> dict[str, str]:
    return {
        "image_path": f"crops/{index}.jpg",
        "body_type": "sedan" if supervised else "unknown",
        "body_type_supervised": "true" if supervised else "false",
        "color": "unknown",
        "color_supervised": "false",
        "split": "train",
        "review_status": "approved",
        "crop_sha256": f"{index:064x}",
        "night": "false",
        "low_light": "false",
        "lighting": "daylight",
        "small_target": "false",
        "vehicle_size": "medium",
        "occluded": "false",
        "truncated": "false",
        "occlusion_level": "visible",
        "hard_mining_tags": "",
        "license_train_eligible": "true",
        "source_dataset": "fixture",
        "source_license": "CC-BY-4.0",
    }


class Stage88ReservoirAuditTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root: Path, candidates: list[dict[str, str]]) -> tuple[Path, Path, Path]:
        base = root / "base.csv"
        base_rows = [row(index, supervised=True) for index in range(10)]
        base_rows[0]["night"] = "true"
        base_rows[1]["vehicle_size"] = "small"
        base_rows[2]["occluded"] = "true"
        self.write_csv(base, base_rows)
        candidate = root / "candidate.csv"
        self.write_csv(candidate, candidates)
        labels = root / "labels.json"
        labels.write_text(json.dumps({"body_types": ["sedan", "unknown"]}), encoding="utf-8")
        return base, candidate, labels

    def test_deduplicates_candidates_and_excludes_base_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [row(0, supervised=False), row(20, supervised=False), row(20, supervised=False)]
            candidates[1]["night"] = "true"
            candidates[2]["night"] = "true"
            base, candidate, labels = self.fixture(root, candidates)
            report = MODULE.audit_reservoir(base, labels, [candidate])
            self.assertEqual(report["unique_licensed_unlabeled_candidates_excluding_base"], 1)
            self.assertEqual(report["eligible_scene_counts"]["night_or_low_light"], 1)

    def test_license_unverified_candidates_do_not_fill_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_row = row(20, supervised=False)
            candidate_row["night"] = "true"
            candidate_row["license_train_eligible"] = "false"
            base, candidate, labels = self.fixture(root, [candidate_row])
            report = MODULE.audit_reservoir(base, labels, [candidate])
            self.assertEqual(report["unique_licensed_unlabeled_candidates_excluding_base"], 0)
            self.assertEqual(report["unique_license_unverified_candidates_excluding_base"], 1)
            self.assertFalse(report["capacity"]["night_or_low_light"]["can_reach_target_even_at_100_percent_acceptance"])

    def test_fails_closed_on_frozen_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_row = row(20, supervised=False)
            candidate_row["image_path"] = "vcas_rtsp_demo_60s/frame.jpg"
            base, candidate, labels = self.fixture(root, [candidate_row])
            with self.assertRaisesRegex(ValueError, "frozen-video"):
                MODULE.audit_reservoir(base, labels, [candidate])


if __name__ == "__main__":
    unittest.main()
