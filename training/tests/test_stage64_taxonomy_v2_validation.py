from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "run_stage64_taxonomy_v2_validation.py"
SPEC = importlib.util.spec_from_file_location("stage64_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage64TaxonomyV2ValidationTest(unittest.TestCase):
    def test_require_sha256_accepts_only_the_pinned_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text("{}", encoding="utf-8")
            expected = MODULE.sha256(path)
            self.assertEqual(MODULE.require_sha256(path, expected.upper(), "evidence"), expected)
            path.write_text("{\"changed\": true}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.require_sha256(path, expected, "evidence")

    def test_require_sha256_rejects_malformed_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.require_sha256(path, "not-a-sha", "evidence")

    def test_threshold_selection_maximizes_coverage_after_precision_gate(self) -> None:
        sweep = {
            "0.5000": {"high_confidence_precision": 0.92, "high_confidence_coverage": 0.8, "high_confidence_selected": 8, "predicted_unknown_rate": 0.1},
            "0.7000": {"high_confidence_precision": 0.94, "high_confidence_coverage": 0.6, "high_confidence_selected": 6, "predicted_unknown_rate": 0.2},
            "0.9000": {"high_confidence_precision": 0.97, "high_confidence_coverage": 0.3, "high_confidence_selected": 3, "predicted_unknown_rate": 0.4},
        }
        selected = MODULE.select_threshold(sweep)
        self.assertEqual(selected["threshold"], 0.7)
        self.assertTrue(selected["precision_gate_pass"])

    def test_validation_views_reject_non_validation_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels.json"
            labels.write_text("{}", encoding="utf-8")
            sources = []
            for name in ("hard", "ua", "vfg"):
                path = root / f"{name}.csv"
                path.write_text("split\nvalidation\n", encoding="utf-8")
                sources.append({"name": name, "output": str(path), "output_sha256": MODULE.sha256(path)})
            report = root / "report.json"
            report.write_text(json.dumps({
                "status": "pass", "labels_sha256": MODULE.sha256(labels), "sources": sources,
                "policy": {
                    "validation_only": False,
                    "merged_or_unsupported_truth_demoted_not_guessed": True,
                    "exact_gray_silver_truth_preserved_when_present": True,
                    "test_accessed": False, "frozen_video_used": False,
                    "production_model_modified": False, "deployment_performed": False,
                },
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.validate_views(report, labels)


if __name__ == "__main__":
    unittest.main()
