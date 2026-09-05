from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_stage107_independent_test_once.py"
SPEC = importlib.util.spec_from_file_location("stage107_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(path: Path, payload: dict) -> tuple[str, str]:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path), sha(path)


class Stage107RunnerTest(unittest.TestCase):
    def test_extracts_only_hash_pinned_validation_selected_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage104 = root / "stage104"
            stage105 = root / "stage105"
            stage104.mkdir()
            stage105.mkdir()
            body = root / "body.pt"
            specialist = root / "specialist.pt"
            color = root / "color.pt"
            for path in (body, specialist, color):
                path.write_bytes(path.name.encode())
            color_path, color_report_sha = report(
                stage104 / "best.json",
                {
                    "status": "complete_validation_only",
                    "policy": {"test_accessed": False, "frozen_video_used": False},
                    "threshold_selection": {"color_shared": {"threshold": 0.84}},
                    "inputs": {
                        "color_checkpoint": str(color),
                        "color_checkpoint_sha256": sha(color),
                    },
                },
            )
            body_path, body_report_sha = report(
                stage105 / "subtype-070.json",
                {
                    "status": "complete_validation_only",
                    "policy": {"test_accessed": False, "frozen_video_used": False},
                    "threshold_selection": {"body_exact": {"threshold": 0.81}},
                    "inputs": {
                        "body_checkpoint": str(body),
                        "body_checkpoint_sha256": sha(body),
                        "body_specialist_checkpoint": str(specialist),
                        "body_specialist_checkpoint_sha256": sha(specialist),
                        "body_specialist_subtype_threshold": 0.7,
                    },
                },
            )
            gate = root / "stage106.json"
            gate.write_text(
                json.dumps(
                    {
                        "status": "ready_for_integrated_independent_test",
                        "authorization": {"independent_test": True},
                        "selected_color": {
                            "variant": "best",
                            "static": {"threshold": 0.84},
                            "report_sha256": color_report_sha,
                        },
                        "selected_body": {
                            "variant": "subtype-070",
                            "subtype_threshold": 0.7,
                            "static": {"threshold": 0.81},
                            "report_sha256": body_report_sha,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE.extract_candidate_inputs(
                gate, sha(gate), stage104, stage105
            )
            self.assertEqual(result["body_checkpoint"], body)
            self.assertEqual(result["body_specialist_checkpoint"], specialist)
            self.assertEqual(result["color_checkpoint"], color)
            self.assertEqual(result["body_threshold"], 0.81)
            self.assertEqual(result["color_threshold"], 0.84)
            self.assertEqual(result["stage104_report"], Path(color_path))
            self.assertEqual(result["stage105_report"], Path(body_path))

    def test_extracts_stage147_per_class_candidate_before_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color_root = root / "color-reports"
            body_root = root / "body-reports"
            (color_root / "gate-best").mkdir(parents=True)
            (body_root / "s288-best").mkdir(parents=True)
            body = root / "body.pt"
            specialist = root / "specialist.pt"
            color = root / "color.pt"
            for path in (body, specialist, color):
                path.write_bytes(path.name.encode())
            body_thresholds = {"bus": 0.843, "heavy_truck": 0.935, "light_truck": 0.986}
            color_thresholds = {"black": 0.887, "white": 0.5, "silver_gray": 1.01}
            body_report = body_root / "s288-best" / "report.json"
            body_report.write_text(
                json.dumps(
                    {
                        "status": "complete_validation_only",
                        "selection": {"thresholds": body_thresholds},
                        "inputs": {"body_specialist_subtype_threshold": 0.5},
                        "policy": {"test_accessed": False, "frozen_video_used": False},
                    }
                ),
                encoding="utf-8",
            )
            color_report = color_root / "gate-best" / "report.json"
            color_report.write_text(
                json.dumps(
                    {
                        "status": "complete_validation_only",
                        "selection": {"thresholds": color_thresholds},
                        "policy": {"test_accessed": False, "frozen_video_used": False},
                    }
                ),
                encoding="utf-8",
            )
            gate = root / "stage147.json"
            gate.write_text(
                json.dumps(
                    {
                        "status": "pass_independent_test_authorized",
                        "independent_test_authorized": True,
                        "candidate": {
                            "body_variant": "s288-best",
                            "body_specialist_checkpoint_sha256": sha(specialist),
                            "body_class_thresholds": body_thresholds,
                            "body_report_sha256": sha(body_report),
                            "color_variant": "gate-best",
                            "color_class_thresholds": color_thresholds,
                            "color_report_sha256": sha(color_report),
                        },
                        "test_accessed": False,
                        "frozen_video_used": False,
                        "production_model_modified": False,
                        "backend_gates_run": False,
                        "deployment_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            result = MODULE.extract_stage147_candidate_inputs(
                gate,
                sha(gate),
                color_root,
                body_root,
                body_checkpoint=body,
                expected_body_checkpoint_sha256=sha(body),
                body_specialist_checkpoint=specialist,
                expected_body_specialist_checkpoint_sha256=sha(specialist),
                color_checkpoint=color,
                expected_color_checkpoint_sha256=sha(color),
            )
            self.assertEqual(result["body_class_thresholds"], body_thresholds)
            self.assertEqual(result["color_class_thresholds"], color_thresholds)
            self.assertEqual(result["body_specialist_subtype_threshold"], 0.5)

    def test_composite_gate_requires_all_named_dataset_gates(self) -> None:
        all_gates = {
            "body_static_precision": True,
            "body_static_coverage": True,
            "body_complex_coverage_gain": True,
            "color_static_precision": True,
            "color_static_coverage": True,
            "color_complex_unknown_reduction": True,
            "body_track_precision": True,
            "body_track_coverage": True,
            "color_track_precision": True,
            "color_track_coverage": True,
            "body_track_stability": True,
            "color_track_stability": True,
        }
        reports = {name: {"gates": dict(all_gates)} for name in ("hard", "vfg", "ua")}
        result = MODULE.composite_gates(reports)
        self.assertTrue(all(result.values()))
        reports["ua"]["gates"]["body_track_stability"] = False
        result = MODULE.composite_gates(reports)
        self.assertFalse(result["ua_body_track_stability"])


if __name__ == "__main__":
    unittest.main()
