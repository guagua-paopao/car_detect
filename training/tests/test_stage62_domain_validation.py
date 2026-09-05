from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage62_domain_validation_after_training.py"
SPEC = importlib.util.spec_from_file_location("stage62_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage62ValidationHelpersTest(unittest.TestCase):
    def test_require_sha256_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text("{}", encoding="utf-8")
            expected = MODULE.sha256(path)
            self.assertEqual(MODULE.require_sha256(path, expected.upper(), "evidence"), expected)
            path.write_text("{\"mutated\": true}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.require_sha256(path, expected, "evidence")

    def test_selects_maximum_coverage_at_precision_gate(self) -> None:
        sweep = {
            "0.7000": {
                "high_confidence_precision": 0.92,
                "high_confidence_coverage": 0.80,
                "high_confidence_selected": 80,
            },
            "0.8000": {
                "high_confidence_precision": 0.94,
                "high_confidence_coverage": 0.60,
                "high_confidence_selected": 60,
            },
            "0.9000": {
                "high_confidence_precision": 0.97,
                "high_confidence_coverage": 0.40,
                "high_confidence_selected": 40,
            },
        }
        chosen = MODULE.select_threshold(sweep)
        self.assertTrue(chosen["precision_gate_pass"])
        self.assertEqual(chosen["threshold"], 0.8)
        self.assertEqual(chosen["coverage"], 0.6)

    def test_reports_failed_precision_gate_without_hiding_best_observation(self) -> None:
        sweep = {
            "0.7000": {
                "high_confidence_precision": 0.80,
                "high_confidence_coverage": 0.90,
                "high_confidence_selected": 90,
            },
            "0.9000": {
                "high_confidence_precision": 0.91,
                "high_confidence_coverage": 0.30,
                "high_confidence_selected": 30,
            },
        }
        chosen = MODULE.select_threshold(sweep)
        self.assertFalse(chosen["precision_gate_pass"])
        self.assertEqual(chosen["precision"], 0.91)

    def test_track_compaction_retains_stability_and_unknown(self) -> None:
        compact = MODULE.compact_track(
            {
                "threshold": 0.9,
                "window": 5,
                "minimum_share": 0.6,
                "minimum_margin": 0.1,
                "final_window": {"precision": 0.94, "coverage": 0.5, "effective_unknown_rate": 0.5},
                "stability": {"weighted_stability_rate": 0.96, "label_switches_total": 2, "abstention_rate": 0.4},
            }
        )
        self.assertEqual(compact["weighted_stability_rate"], 0.96)
        self.assertEqual(compact["effective_unknown_rate"], 0.5)
        self.assertEqual(compact["label_switches_total"], 2)

    def test_release_gates_do_not_fabricate_ua_color_truth(self) -> None:
        gates = MODULE.build_release_gates(
            hard_body={"precision": 0.94, "coverage": 0.50},
            hard_color={"precision": 0.95, "coverage": 0.30},
            ua_body={"precision": 0.94, "coverage": 0.50, "weighted_stability_rate": 0.96},
            vfg_screen_passed=True,
            body_gain=0.16,
            color_unknown_reduction=0.21,
            is_baseline=False,
        )
        self.assertTrue(all(gates.values()))
        self.assertNotIn("ua_color_precision_0_93", gates)
        self.assertNotIn("ua_color_coverage_0_25", gates)
        self.assertIn("vfg_labeled_body_color_and_track_screen", gates)

    def test_compact_track_records_zero_evaluated_truth(self) -> None:
        compact = MODULE.compact_track(
            {
                "threshold": 0.8,
                "window": 3,
                "minimum_share": 0.5,
                "minimum_margin": 0.05,
                "final_window": {"evaluated": 0, "precision": 0.0, "coverage": 0.0},
                "stability": {"weighted_stability_rate": 1.0},
            }
        )
        self.assertEqual(compact["evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
