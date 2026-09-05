from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_stage105_body_subtype_sweep.py"
SPEC = importlib.util.spec_from_file_location("summarize_stage105", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(threshold: float, coverage: float, gain: float, all_pass: bool) -> dict[str, object]:
    gates = {key: all_pass for key in MODULE.BODY_GATES}
    if not all_pass:
        gates["body_static_precision"] = True
        gates["body_track_precision"] = True
        gates["body_track_stability"] = True
    return {
        "status": "complete_validation_only",
        "policy": {"test_accessed": False, "frozen_video_used": False},
        "inputs": {"body_specialist_subtype_threshold": threshold},
        "shared_validation_gates": {"gates": gates},
        "threshold_selection": {"body_exact": {"precision": 0.93, "coverage": coverage}},
        "comparison": {"body_complex_static_coverage_gain": gain},
        "track_fusion": {"candidate": {"body_exact": {
            "track_final": {"precision": 0.94, "coverage": coverage},
            "stability": {"transition_stability": 0.99},
        }}},
    }


class Stage105SummaryTest(unittest.TestCase):
    def test_fail_closed_but_selects_best_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "subtype-070.json").write_text(json.dumps(report(0.7, 0.25, 0.13, False)), encoding="utf-8")
            (root / "subtype-080.json").write_text(json.dumps(report(0.8, 0.27, 0.14, False)), encoding="utf-8")
            result = MODULE.summarize(root)
            self.assertEqual(result["decision"], "threshold_only_repair_rejected_fail_closed")
            self.assertEqual(result["selected_diagnostic"], "subtype-080")

    def test_qualified_variant_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "subtype-070.json").write_text(json.dumps(report(0.7, 0.45, 0.16, True)), encoding="utf-8")
            (root / "subtype-080.json").write_text(json.dumps(report(0.8, 0.30, 0.14, False)), encoding="utf-8")
            result = MODULE.summarize(root)
            self.assertEqual(result["qualified_variants"], ["subtype-070"])
            self.assertEqual(result["selected_diagnostic"], "subtype-070")


if __name__ == "__main__":
    unittest.main()
