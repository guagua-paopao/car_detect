from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_v2_decoupled_shared_test_once.py"
SPEC = importlib.util.spec_from_file_location("stage107_test_once", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage106_state() -> dict:
    return {
        "status": "ready_for_integrated_independent_test",
        "selected_body": {
            "variant": "subtype-070",
            "subtype_threshold": 0.7,
            "static": {"threshold": 0.81},
        },
        "selected_color": {
            "variant": "best",
            "static": {"threshold": 0.84},
        },
        "failure_reasons": [],
        "authorization": {
            "independent_test": True,
            "onnx_backend_gates": False,
            "frozen_video_replay": False,
            "deployment": False,
        },
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def stage147_state() -> dict:
    return {
        "status": "pass_independent_test_authorized",
        "independent_test_authorized": True,
        "candidate": {
            "body_variant": "s288-best",
            "body_specialist_checkpoint_sha256": "a" * 64,
            "body_class_thresholds": {
                "bus": 0.843,
                "heavy_truck": 0.935,
                "light_truck": 0.986,
                "other": 1.01,
            },
            "color_variant": "gate-best",
            "color_class_thresholds": {
                "black": 0.887,
                "white": 0.5,
                "silver_gray": 1.01,
            },
        },
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


class Stage107TestOnceTest(unittest.TestCase):
    def test_parameters_are_loaded_only_from_safe_stage106_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage106.json"
            path.write_text(json.dumps(stage106_state()), encoding="utf-8")
            result = MODULE.load_component_gate_parameters(path, digest(path))
        self.assertEqual(result["body_thresholds"], 0.81)
        self.assertEqual(result["color_thresholds"], 0.84)
        self.assertEqual(result["body_specialist_subtype_threshold"], 0.7)

    def test_stage147_per_class_thresholds_are_loaded_without_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage147.json"
            path.write_text(json.dumps(stage147_state()), encoding="utf-8")
            result = MODULE.load_component_gate_parameters(path, digest(path))
        self.assertEqual(result["body_thresholds"]["light_truck"], 0.986)
        self.assertEqual(result["color_thresholds"]["white"], 0.5)
        self.assertEqual(result["body_specialist_subtype_threshold"], 0.5)
        self.assertEqual(result["threshold_policy"], "validation_selected_per_class")

    def test_unsafe_stage106_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage106.json"
            state = stage106_state()
            state["authorization"]["frozen_video_replay"] = True
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen_video_replay"):
                MODULE.load_component_gate_parameters(path, digest(path))

    def test_unsafe_stage147_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage147.json"
            state = stage147_state()
            state["test_accessed"] = True
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "test_accessed"):
                MODULE.load_component_gate_parameters(path, digest(path))

    def test_test_manifest_must_be_approved_safe_and_test_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            fields = ["image_path", "split", "review_status", "source_frame_id"]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": image.name,
                        "split": "test",
                        "review_status": "approved",
                        "source_frame_id": "safe",
                    }
                )
            rows = MODULE.resolve_test_rows(manifest, root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["_resolved_image_path"], str(image.resolve()))

            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": image.name,
                        "split": "validation",
                        "review_status": "approved",
                        "source_frame_id": "safe",
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "test-only"):
                MODULE.resolve_test_rows(manifest, root)

    def test_relative_unknown_reduction(self) -> None:
        self.assertAlmostEqual(MODULE.relative_unknown_reduction(0.8, 0.6), 0.25)
        self.assertEqual(MODULE.relative_unknown_reduction(0.0, 0.0), 0.0)

    def test_metric_bundle_applies_fixed_per_class_thresholds(self) -> None:
        rows = [
            {
                "body_type_supervised": "true",
                "body_type": "bus",
                "track_key": "track-1",
                "lighting": "night",
                "vehicle_size": "large",
            },
            {
                "body_type_supervised": "true",
                "body_type": "heavy_truck",
                "track_key": "track-2",
                "lighting": "night",
                "vehicle_size": "large",
            },
        ]
        outputs = [
            {"body_label": "bus", "body_confidence": 0.85},
            {"body_label": "heavy_truck", "body_confidence": 0.90},
        ]
        bundle = MODULE.metric_bundle(
            rows, outputs, "body", {"bus": 0.843, "heavy_truck": 0.935}
        )
        self.assertEqual(bundle["static"]["selected"], 1)
        self.assertEqual(bundle["static"]["correct"], 1)
        self.assertEqual(bundle["static"]["coverage"], 0.5)
        self.assertEqual(bundle["track"]["track_final"]["selected"], 1)


if __name__ == "__main__":
    unittest.main()
