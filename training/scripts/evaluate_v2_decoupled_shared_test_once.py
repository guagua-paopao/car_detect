#!/usr/bin/env python3
"""Evaluate one validation-authorized v2 body/color pair on test exactly once.

Thresholds and the truck-specialist routing threshold are read from immutable
component-gate evidence.  This evaluator never searches test parameters,
never reads the frozen demo video, and never mutates production artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_v2_decoupled_shared_validation as validation  # noqa: E402
import evaluate_stage109_color_class_thresholds as color_thresholding  # noqa: E402
import evaluate_stage110_body_class_thresholds as body_thresholding  # noqa: E402


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_component_gate_parameters(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_sha(path, expected_sha256, "component gate")
    state = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "test_accessed",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if state.get(key) is not False:
            raise RuntimeError(f"unsafe component-gate state: {key}")
    if state.get("status") == "pass_independent_test_authorized":
        if state.get("independent_test_authorized") is not True:
            raise RuntimeError("component gate did not authorize independent test")
        candidate = state.get("candidate")
        if not isinstance(candidate, dict):
            raise RuntimeError("component gate candidate is missing")
        body_thresholds = candidate.get("body_class_thresholds")
        color_thresholds = candidate.get("color_class_thresholds")
        if not isinstance(body_thresholds, dict) or not isinstance(color_thresholds, dict):
            raise RuntimeError("component gate class thresholds are missing")
        for role, thresholds in (("body", body_thresholds), ("color", color_thresholds)):
            if not thresholds or any(not 0.0 < float(value) <= 1.01 for value in thresholds.values()):
                raise RuntimeError(f"invalid validation-selected {role} class thresholds")
        return {
            "body_thresholds": {key: float(value) for key, value in body_thresholds.items()},
            "color_thresholds": {key: float(value) for key, value in color_thresholds.items()},
            "body_specialist_subtype_threshold": 0.50,
            "selected_body_variant": candidate["body_variant"],
            "selected_color_variant": candidate["color_variant"],
            "expected_body_specialist_checkpoint_sha256": candidate[
                "body_specialist_checkpoint_sha256"
            ],
            "component_gate_state_sha256": sha256(path),
            "threshold_policy": "validation_selected_per_class",
        }
    if state.get("status") != "ready_for_integrated_independent_test":
        raise RuntimeError("component gate did not authorize the integrated independent test")
    authorization = state.get("authorization", {})
    if authorization.get("independent_test") is not True:
        raise RuntimeError("component gate independent-test authorization is missing")
    for key in ("onnx_backend_gates", "frozen_video_replay", "deployment"):
        if authorization.get(key) is not False:
            raise RuntimeError(f"unsafe component-gate authorization: {key}")
    body = state.get("selected_body")
    color = state.get("selected_color")
    if not isinstance(body, dict) or not isinstance(color, dict):
        raise RuntimeError("selected component evidence is incomplete")
    return {
        "body_thresholds": float(body["static"]["threshold"]),
        "color_thresholds": float(color["static"]["threshold"]),
        "body_specialist_subtype_threshold": float(body["subtype_threshold"]),
        "selected_body_variant": body["variant"],
        "selected_color_variant": color["variant"],
        "component_gate_state_sha256": sha256(path),
        "threshold_policy": "validation_selected_global",
    }


# Backward-compatible import used by the immutable Stage107 test-view builder.
load_stage106_parameters = load_component_gate_parameters


def resolve_test_rows(manifest: Path, safety_root: Path) -> list[dict[str, str]]:
    safety_root = safety_root.resolve()
    if not safety_root.is_dir():
        raise RuntimeError("datasets safety root is missing")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("test manifest has no rows")
    for line, row in enumerate(rows, start=2):
        if row.get("split") != "test":
            raise RuntimeError(f"test-only manifest contains split={row.get('split')} at line {line}")
        searchable = " ".join(str(value).lower() for value in row.values())
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"frozen marker in test row {line}")
        if row.get("review_status") != "approved":
            raise RuntimeError(f"unapproved test row {line}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"test image escapes datasets safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["_resolved_image_path"] = str(resolved)
    return rows


def head_indexes(rows: list[dict[str, str]], head: str) -> list[int]:
    field = "body_type_supervised" if head == "body" else "color_supervised"
    return [index for index, row in enumerate(rows) if truthy(row.get(field))]


def metric_bundle(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    threshold_policy: float | dict[str, float],
) -> dict[str, Any]:
    indexes = head_indexes(rows, head)
    truth_field = "body_type" if head == "body" else "color"
    predictions = [outputs[index][f"{head}_label"] for index in indexes]
    confidences = [outputs[index][f"{head}_confidence"] for index in indexes]
    truths = [rows[index][truth_field] for index in indexes]
    complex_indexes = [index for index in indexes if validation.is_complex_row(rows[index])]
    complex_predictions = [outputs[index][f"{head}_label"] for index in complex_indexes]
    complex_confidences = [outputs[index][f"{head}_confidence"] for index in complex_indexes]
    complex_truths = [rows[index][truth_field] for index in complex_indexes]
    if isinstance(threshold_policy, dict):
        thresholds = threshold_policy
    else:
        labels = {output[f"{head}_label"] for output in outputs}
        thresholds = {label: float(threshold_policy) for label in labels if label != "unknown"}
    stratified = (
        body_thresholding.stratified(rows, outputs, thresholds)
        if head == "body"
        else color_thresholding.stratified(rows, outputs, thresholds)
    )
    track = (
        body_thresholding.fused(rows, outputs, thresholds)
        if head == "body"
        else color_thresholding.fused(rows, outputs, thresholds)
    )
    return {
        "supervised_rows": len(indexes),
        "complex_supervised_rows": len(complex_indexes),
        "static": color_thresholding.metric(predictions, confidences, truths, thresholds),
        "complex_static": color_thresholding.metric(
            complex_predictions, complex_confidences, complex_truths, thresholds
        ),
        "per_class": color_thresholding.per_class_metrics(
            predictions, confidences, truths, thresholds
        ),
        "stratified": stratified,
        "track": track,
    }


def relative_unknown_reduction(baseline_unknown: float, candidate_unknown: float) -> float:
    return (
        (baseline_unknown - candidate_unknown) / baseline_unknown
        if baseline_unknown > 0.0
        else 0.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization-state", "--stage106-state",
        dest="authorization_state", type=Path, required=True,
    )
    parser.add_argument(
        "--expected-authorization-state-sha256", "--expected-stage106-state-sha256",
        dest="expected_authorization_state_sha256", required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--body-specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-specialist-checkpoint-sha256", required=True)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-color-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--production-body-threshold", type=float, default=0.75)
    parser.add_argument("--production-color-threshold", type=float, default=0.70)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite one-time test evidence: {args.output}")
    pinned = (
        (args.manifest, args.expected_manifest_sha256, "test manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256, "body checkpoint"),
        (
            args.body_specialist_checkpoint,
            args.expected_body_specialist_checkpoint_sha256,
            "body specialist checkpoint",
        ),
        (args.color_checkpoint, args.expected_color_checkpoint_sha256, "color checkpoint"),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "baseline checkpoint"),
    )
    for path, expected, label in pinned:
        require_sha(path, expected, label)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 labels are required")
    parameters = load_component_gate_parameters(
        args.authorization_state, args.expected_authorization_state_sha256
    )
    expected_specialist = parameters.get("expected_body_specialist_checkpoint_sha256")
    if expected_specialist and sha256(args.body_specialist_checkpoint) != expected_specialist:
        raise RuntimeError("authorized specialist checkpoint does not match evaluator input")
    args.body_specialist_subtype_threshold = parameters[
        "body_specialist_subtype_threshold"
    ]
    rows = resolve_test_rows(args.manifest, args.datasets_safety_root)
    candidate = validation.infer_pair(
        args,
        rows,
        args.body_checkpoint,
        args.color_checkpoint,
        expected_labels=labels,
        body_specialist_checkpoint_path=args.body_specialist_checkpoint,
    )
    baseline = validation.infer_pair(
        args,
        rows,
        args.baseline_checkpoint,
        args.baseline_checkpoint,
        expected_labels=None,
    )
    candidate_body = metric_bundle(
        rows, candidate, "body", parameters["body_thresholds"]
    )
    candidate_color = metric_bundle(
        rows, candidate, "color", parameters["color_thresholds"]
    )
    baseline_body = metric_bundle(
        rows, baseline, "body", args.production_body_threshold
    )
    baseline_color = metric_bundle(
        rows, baseline, "color", args.production_color_threshold
    )
    body_gain = (
        candidate_body["complex_static"]["coverage"]
        - baseline_body["complex_static"]["coverage"]
    )
    color_unknown_reduction = relative_unknown_reduction(
        baseline_color["complex_static"]["unknown_rate"],
        candidate_color["complex_static"]["unknown_rate"],
    )
    gates = {
        "body_static_precision": candidate_body["static"]["precision"] >= 0.93,
        "body_static_coverage": candidate_body["static"]["coverage"] >= 0.45,
        "body_complex_coverage_gain": body_gain >= 0.15,
        "color_static_precision": candidate_color["static"]["precision"] >= 0.93,
        "color_static_coverage": candidate_color["static"]["coverage"] >= 0.25,
        "color_complex_unknown_reduction": color_unknown_reduction >= 0.20,
        "body_track_precision": candidate_body["track"]["track_final"]["precision"] >= 0.93,
        "body_track_coverage": candidate_body["track"]["track_final"]["coverage"] >= 0.45,
        "color_track_precision": candidate_color["track"]["track_final"]["precision"] >= 0.93,
        "color_track_coverage": candidate_color["track"]["track_final"]["coverage"] >= 0.25,
        "body_track_stability": candidate_body["track"]["stability"]["transition_stability"] >= 0.95,
        "color_track_stability": candidate_color["track"]["stability"]["transition_stability"] >= 0.95,
    }
    # Re-pin all mutable external inputs after inference.
    for path, expected, label in pinned:
        require_sha(path, expected, label)
    require_sha(
        args.authorization_state,
        args.expected_authorization_state_sha256,
        "component gate",
    )
    report = {
        "schema_version": "v2-decoupled-shared-test-once-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_test_once",
        "decision": "test_gates_pass" if all(gates.values()) else "test_gates_fail_closed",
        "inputs": {
            "authorization_state": str(args.authorization_state.resolve()),
            "authorization_state_sha256": sha256(args.authorization_state),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "labels_sha256": sha256(args.labels),
            "body_checkpoint": str(args.body_checkpoint.resolve()),
            "body_checkpoint_sha256": sha256(args.body_checkpoint),
            "body_specialist_checkpoint": str(args.body_specialist_checkpoint.resolve()),
            "body_specialist_checkpoint_sha256": sha256(args.body_specialist_checkpoint),
            "color_checkpoint": str(args.color_checkpoint.resolve()),
            "color_checkpoint_sha256": sha256(args.color_checkpoint),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        },
        "fixed_validation_parameters": {
            **parameters,
            "production_body_threshold": args.production_body_threshold,
            "production_color_threshold": args.production_color_threshold,
            "parameter_search_on_test": False,
        },
        "rows": len(rows),
        "candidate": {"body": candidate_body, "color": candidate_color},
        "production_baseline": {"body": baseline_body, "color": baseline_color},
        "comparison": {
            "body_complex_static_coverage_gain": body_gain,
            "color_complex_static_unknown_relative_reduction": color_unknown_reduction,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "policy": {
            "split": "test",
            "test_accessed": True,
            "test_used_for_selection": False,
            "threshold_or_temperature_search": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "gates": gates,
                "test_used_for_selection": False,
                "frozen_video_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
