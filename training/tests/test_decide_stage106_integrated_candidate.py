from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "decide_stage106_integrated_candidate.py"
SPEC = importlib.util.spec_from_file_location("stage106", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def safe_fields() -> dict[str, bool]:
    return {
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def color_state(passing: bool = True) -> dict:
    gates = {key: passing for key in MODULE.COLOR_GATES}
    return {
        "status": "complete_validation_only",
        "decision": "color_component_qualified" if passing else "color_candidate_rejected_fail_closed",
        "qualified_variants": ["best"] if passing else [],
        "variants": {
            "best": {
                "static": {"precision": 0.94, "coverage": 0.31},
                "complex_unknown_relative_reduction": 0.22,
                "track_final": {"precision": 0.94, "coverage": 0.30},
                "stability": 0.98,
                "color_gates": gates,
                "all_color_gates_pass": passing,
            }
        },
        **safe_fields(),
    }


def body_state(passing: bool = True) -> dict:
    gates = {key: passing for key in MODULE.BODY_GATES}
    return {
        "status": "complete_validation_only",
        "decision": "body_subtype_threshold_qualified" if passing else "threshold_only_repair_rejected_fail_closed",
        "qualified_variants": ["subtype-070"] if passing else [],
        "variants": [
            {
                "variant": "subtype-070",
                "subtype_threshold": 0.7,
                "static": {"precision": 0.94, "coverage": 0.46},
                "complex_coverage_gain": 0.16,
                "track_final": {"precision": 0.94, "coverage": 0.46},
                "stability": 0.98,
                "body_gates": gates,
                "all_body_gates_pass": passing,
            }
        ],
        **safe_fields(),
    }


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Stage106DecisionTest(unittest.TestCase):
    def test_all_components_pass_only_authorizes_independent_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = MODULE.decide(
                write(root / "color.json", color_state()),
                write(root / "body.json", body_state()),
            )
        self.assertEqual(result["status"], "ready_for_integrated_independent_test")
        self.assertEqual(
            result["authorization"],
            {
                "independent_test": True,
                "onnx_backend_gates": False,
                "frozen_video_replay": False,
                "deployment": False,
            },
        )
        self.assertEqual(result["failure_reasons"], [])

    def test_component_failure_closes_gate(self) -> None:
        for failed in ("color", "body"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = MODULE.decide(
                    write(root / "color.json", color_state(failed != "color")),
                    write(root / "body.json", body_state(failed != "body")),
                )
                self.assertEqual(result["status"], "component_repair_required_fail_closed")
                self.assertFalse(result["authorization"]["independent_test"])
                self.assertFalse(result["authorization"]["frozen_video_replay"])

    def test_unsafe_upstream_state_is_rejected(self) -> None:
        unsafe = color_state()
        unsafe["frozen_video_used"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "frozen_video_used"):
                MODULE.decide(
                    write(root / "color.json", unsafe),
                    write(root / "body.json", body_state()),
                )

    def test_claimed_qualification_requires_all_named_gates(self) -> None:
        inconsistent = color_state()
        inconsistent["variants"]["best"]["color_gates"]["color_track_stability"] = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "fully passing variant"):
                MODULE.decide(
                    write(root / "color.json", inconsistent),
                    write(root / "body.json", body_state()),
                )


if __name__ == "__main__":
    unittest.main()
