from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_stage104_color_validation.py"
SPEC = importlib.util.spec_from_file_location("summarize_stage104", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(pass_unknown: bool = True) -> dict[str, object]:
    gates = {key: True for key in MODULE.COLOR_GATES}
    gates["color_complex_unknown_reduction"] = pass_unknown
    return {
        "status": "complete_validation_only",
        "policy": {"test_accessed": False, "frozen_video_used": False},
        "shared_validation_gates": {"gates": gates},
        "threshold_selection": {"color_shared": {"precision": 0.93, "coverage": 0.3}},
        "comparison": {"color_complex_static_unknown_relative_reduction": 0.21},
        "track_fusion": {"candidate": {"color_shared": {
            "track_final": {"precision": 0.94, "coverage": 0.3},
            "stability": {"transition_stability": 0.99},
        }}},
    }


class Stage104SummaryTest(unittest.TestCase):
    def test_qualifies_only_variants_passing_all_color_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "best.json").write_text(json.dumps(report(True)), encoding="utf-8")
            (root / "gate-best.json").write_text(json.dumps(report(False)), encoding="utf-8")
            result = MODULE.summarize(root)
            self.assertEqual(result["decision"], "color_component_qualified")
            self.assertEqual(result["qualified_variants"], ["best"])

    def test_rejects_report_that_accessed_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsafe = report(True)
            unsafe["policy"]["test_accessed"] = True
            (root / "best.json").write_text(json.dumps(unsafe), encoding="utf-8")
            (root / "gate-best.json").write_text(json.dumps(report(True)), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.summarize(root)


if __name__ == "__main__":
    unittest.main()
