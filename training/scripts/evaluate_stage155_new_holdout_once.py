#!/usr/bin/env python3
"""One-time evaluation of the fixed Stage154 candidate on the new Stage155 holdout."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import evaluate_v2_decoupled_shared_validation as base


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated-state", type=Path, required=True)
    parser.add_argument("--expected-integrated-state-sha256", required=True)
    parser.add_argument("--holdout-state", type=Path, required=True)
    parser.add_argument("--expected-holdout-state-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-body-threshold", type=float, required=True)
    parser.add_argument("--baseline-color-threshold", type=float, required=True)
    parser.add_argument("--body-specialist-subtype-threshold", type=float, default=0.50)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256(path)
    if actual != expected.lower():
        raise RuntimeError(f"{role} SHA mismatch: expected={expected} actual={actual}")
    return actual


def emitted(prediction: str, confidence: float, thresholds: dict[str, float] | float) -> str:
    threshold = thresholds.get(prediction, 1.01) if isinstance(thresholds, dict) else thresholds
    return prediction if prediction != "unknown" and confidence >= threshold else "unknown"


def metric(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    indexes: list[int],
    head: str,
    thresholds: dict[str, float] | float,
) -> dict[str, Any]:
    selected = []
    correct = 0
    truth_field = "body_type" if head == "body" else "color"
    for index in indexes:
        prediction = emitted(outputs[index][f"{head}_label"], outputs[index][f"{head}_confidence"], thresholds)
        if prediction != "unknown":
            selected.append(index)
            correct += prediction == rows[index][truth_field]
    evaluated = len(indexes)
    return {
        "evaluated": evaluated, "selected": len(selected), "correct": correct,
        "precision": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / evaluated if evaluated else 0.0,
        "unknown_rate": 1.0 - len(selected) / evaluated if evaluated else 1.0,
    }


def per_class(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    indexes: list[int],
    head: str,
    thresholds: dict[str, float] | float,
) -> dict[str, Any]:
    truth_field = "body_type" if head == "body" else "color"
    result = {}
    for label in sorted({rows[index][truth_field] for index in indexes}):
        subset = [index for index in indexes if rows[index][truth_field] == label]
        result[label] = metric(rows, outputs, subset, head, thresholds)
    return result


def stratified(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    indexes: list[int],
    head: str,
    thresholds: dict[str, float] | float,
) -> dict[str, Any]:
    result = {}
    for field in ("source_dataset", "lighting", "vehicle_size", "camera_id", "video_id"):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index in indexes:
            groups[rows[index].get(field) or "unknown"].append(index)
        result[field] = {key: metric(rows, outputs, subset, head, thresholds) for key, subset in sorted(groups.items())}
    return result


def track_metric(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    thresholds: dict[str, float] | float,
) -> dict[str, Any]:
    supervised_field = "body_type_supervised" if head == "body" else "color_supervised"
    truth_field = "body_type" if head == "body" else "color"
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("source_dataset") == "Vehicle-Rear" and base.truthy(row.get(supervised_field)):
            groups[row["track_group"]].append(index)
    groups = defaultdict(list, {key: value for key, value in groups.items() if len(value) >= 2})
    final_predictions: list[str] = []
    final_truths: list[str] = []
    labels_by_track: dict[str, list[str]] = {}
    for track, indexes in groups.items():
        observations = []
        for index in indexes:
            prediction = emitted(outputs[index][f"{head}_label"], outputs[index][f"{head}_confidence"], thresholds)
            observations.append((prediction, outputs[index][f"{head}_confidence"], base.quality_weight(rows[index])))
        fused = base.fuse_observations(observations)
        labels_by_track[track] = fused
        final_predictions.append(fused[-1])
        final_truths.append(Counter(rows[index][truth_field] for index in indexes).most_common(1)[0][0])
    selected = [index for index, label in enumerate(final_predictions) if label != "unknown"]
    correct = sum(final_predictions[index] == final_truths[index] for index in selected)
    total = len(final_truths)
    return {
        "track_final": {"evaluated": total, "selected": len(selected), "correct": correct,
                        "precision": correct / len(selected) if selected else 0.0,
                        "coverage": len(selected) / total if total else 0.0,
                        "unknown_rate": 1.0 - len(selected) / total if total else 1.0},
        "stability": base.stability(labels_by_track), "window": 5, "minimum_share": 0.60, "minimum_margin": 0.15,
    }


def relative_unknown_reduction(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    denominator = baseline["unknown_rate"]
    return (denominator - candidate["unknown_rate"]) / denominator if denominator > 0 else 0.0


def resolve_test_rows(manifest: Path, safety_root: Path) -> list[dict[str, str]]:
    """Resolve only the sealed Stage155 test view without weakening validation parsing."""
    safety_root = safety_root.resolve()
    if not safety_root.is_dir():
        raise RuntimeError("datasets safety root is missing")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Stage155 test manifest has no rows")
    for line, row in enumerate(rows, start=2):
        if row.get("split") != "test":
            raise RuntimeError(f"Stage155 test-only manifest contains split={row.get('split')} at line {line}")
        searchable = " ".join(row.values()).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"frozen marker in Stage155 test row {line}")
        if row.get("review_status") != "approved":
            raise RuntimeError(f"unapproved Stage155 test row {line}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"Stage155 test image escapes datasets safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["_resolved_image_path"] = str(resolved)
    return rows


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.state_output.exists():
        raise FileExistsError("refusing to overwrite Stage155 one-time evidence")
    for path, expected, role in (
        (args.integrated_state, args.expected_integrated_state_sha256, "Stage154 integrated state"),
        (args.holdout_state, args.expected_holdout_state_sha256, "Stage155 holdout state"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "production baseline checkpoint"),
    ):
        assert_hash(path, expected, role)
    integrated = json.loads(args.integrated_state.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_state.read_text(encoding="utf-8"))
    if integrated.get("status") != "pass_stage155_new_holdout_authorized" or not integrated.get("stage155_new_holdout_authorized"):
        raise RuntimeError("Stage154 did not authorize Stage155")
    if holdout.get("status") != "holdout_built_pending_one_time_evaluation":
        raise RuntimeError("Stage155 holdout is not evaluation-ready")
    for state in (integrated, holdout):
        for key in ("frozen_video_used", "production_model_modified", "backend_gates_run", "deployment_performed"):
            if state.get(key) is not False:
                raise RuntimeError(f"isolation policy failed: {key}")
    candidate = integrated["candidate"]
    body_path = Path(candidate["body_primary_checkpoint"])
    specialist_path = Path(candidate["body_specialist_checkpoint"])
    color_path = Path(candidate["color_checkpoint"])
    assert_hash(body_path, candidate["body_primary_checkpoint_sha256"], "candidate body checkpoint")
    assert_hash(specialist_path, candidate["body_specialist_checkpoint_sha256"], "candidate body specialist")
    assert_hash(color_path, candidate["color_checkpoint_sha256"], "candidate color checkpoint")
    manifest = Path(holdout["manifest"])
    assert_hash(manifest, holdout["manifest_sha256"], "Stage155 manifest")
    if any(marker in " ".join(map(str, (manifest, body_path, specialist_path, color_path))).lower() for marker in FROZEN_MARKERS):
        raise RuntimeError("frozen marker reached Stage155 evaluator")
    rows = resolve_test_rows(manifest, args.datasets_safety_root)
    if not rows or any(row.get("split") != "test" for row in rows):
        raise RuntimeError("Stage155 manifest is not a non-empty test-only view")
    if any(not base.truthy(row.get("research_only")) or base.truthy(row.get("deployment_eligible")) for row in rows):
        raise RuntimeError("Stage155 research isolation failed")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    candidate_outputs = base.infer_pair(args, rows, body_path, color_path, expected_labels=labels, body_specialist_checkpoint_path=specialist_path)
    baseline_outputs = base.infer_pair(args, rows, args.baseline_checkpoint, args.baseline_checkpoint, expected_labels=None)
    body_thresholds = {key: float(value) for key, value in candidate["body_class_thresholds"].items()}
    color_thresholds = {key: float(value) for key, value in candidate["color_class_thresholds"].items()}
    body_indexes = [index for index, row in enumerate(rows) if base.truthy(row.get("body_type_supervised"))]
    color_indexes = [index for index, row in enumerate(rows) if base.truthy(row.get("color_supervised"))]
    if len(body_indexes) < 300 or len(color_indexes) < 650:
        raise RuntimeError(f"insufficient Stage155 supervision: body={len(body_indexes)} color={len(color_indexes)}")
    body_candidate = metric(rows, candidate_outputs, body_indexes, "body", body_thresholds)
    body_baseline = metric(rows, baseline_outputs, body_indexes, "body", args.baseline_body_threshold)
    color_candidate = metric(rows, candidate_outputs, color_indexes, "color", color_thresholds)
    color_baseline = metric(rows, baseline_outputs, color_indexes, "color", args.baseline_color_threshold)
    body_complex_indexes = [index for index in body_indexes if base.is_complex_row(rows[index])]
    color_complex_indexes = [index for index in color_indexes if base.is_complex_row(rows[index])]
    if len(body_complex_indexes) < 100 or len(color_complex_indexes) < 50:
        raise RuntimeError(f"insufficient complex Stage155 supervision: body={len(body_complex_indexes)} color={len(color_complex_indexes)}")
    body_complex_candidate = metric(rows, candidate_outputs, body_complex_indexes, "body", body_thresholds)
    body_complex_baseline = metric(rows, baseline_outputs, body_complex_indexes, "body", args.baseline_body_threshold)
    color_complex_candidate = metric(rows, candidate_outputs, color_complex_indexes, "color", color_thresholds)
    color_complex_baseline = metric(rows, baseline_outputs, color_complex_indexes, "color", args.baseline_color_threshold)
    body_gain = body_complex_candidate["coverage"] - body_complex_baseline["coverage"]
    color_unknown_reduction = relative_unknown_reduction(color_complex_candidate, color_complex_baseline)
    body_track = track_metric(rows, candidate_outputs, "body", body_thresholds)
    color_track = track_metric(rows, candidate_outputs, "color", color_thresholds)
    body_track_final, color_track_final = body_track["track_final"], color_track["track_final"]
    body_stability, color_stability = body_track["stability"], color_track["stability"]
    gates = {
        "body_static_precision": body_candidate["precision"] >= 0.93,
        "body_static_coverage": body_candidate["coverage"] >= 0.45,
        "body_complex_coverage_gain": body_gain >= 0.15,
        "color_static_precision": color_candidate["precision"] >= 0.93,
        "color_static_coverage": color_candidate["coverage"] >= 0.25,
        "color_unknown_relative_reduction": color_unknown_reduction >= 0.20,
        "body_track_nonempty": body_track_final["evaluated"] >= 20,
        "body_track_precision": body_track_final["precision"] >= 0.93,
        "body_track_coverage": body_track_final["coverage"] >= 0.25,
        "body_track_stability": body_stability["transition_stability"] >= 0.95,
        "color_track_nonempty": color_track_final["evaluated"] >= 100,
        "color_track_precision": color_track_final["precision"] >= 0.93,
        "color_track_coverage": color_track_final["coverage"] >= 0.25,
        "color_track_stability": color_stability["transition_stability"] >= 0.95,
    }
    passed = all(gates.values())
    report = {
        "schema_version": "stage155-new-unused-holdout-evaluation-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass_stage156_backend_authorized" if passed else "candidate_rejected_fail_closed_before_backend",
        "inputs": {
            "integrated_state": str(args.integrated_state), "integrated_state_sha256": sha256(args.integrated_state),
            "holdout_state": str(args.holdout_state), "holdout_state_sha256": sha256(args.holdout_state),
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "body_checkpoint": str(body_path), "body_checkpoint_sha256": sha256(body_path),
            "specialist_checkpoint": str(specialist_path), "specialist_checkpoint_sha256": sha256(specialist_path),
            "color_checkpoint": str(color_path), "color_checkpoint_sha256": sha256(color_path),
            "baseline_checkpoint": str(args.baseline_checkpoint), "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        },
        "threshold_contract": {
            "candidate_body": body_thresholds, "candidate_color": color_thresholds,
            "production_body": args.baseline_body_threshold, "production_color": args.baseline_color_threshold,
            "threshold_search_performed": False, "model_selection_performed": False,
        },
        "body": {
            "candidate": body_candidate, "production_baseline": body_baseline, "complex_coverage_gain": body_gain,
            "complex": {"candidate": body_complex_candidate, "production_baseline": body_complex_baseline},
            "per_class": per_class(rows, candidate_outputs, body_indexes, "body", body_thresholds),
            "stratified": stratified(rows, candidate_outputs, body_indexes, "body", body_thresholds), "track": body_track,
        },
        "color": {
            "candidate": color_candidate, "production_baseline": color_baseline,
            "unknown_relative_reduction": color_unknown_reduction,
            "complex": {"candidate": color_complex_candidate, "production_baseline": color_complex_baseline},
            "per_class": per_class(rows, candidate_outputs, color_indexes, "color", color_thresholds),
            "stratified": stratified(rows, candidate_outputs, color_indexes, "color", color_thresholds), "track": color_track,
        },
        "gates": gates,
        "policy": {"test_accessed": True, "test_used_for_selection": False, "stage148_test_reused": False,
                   "frozen_video_used": False, "production_model_modified": False,
                   "backend_gates_run": False, "deployment_performed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": "stage155-new-unused-holdout-evaluation-state-v1", "created_at": report["created_at"],
        "status": report["decision"], "stage156_backend_authorized": passed,
        "report": str(args.output), "report_sha256": sha256(args.output), "gates": gates,
        "candidate": candidate, **report["policy"],
    }
    args.state_output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    Path(str(args.state_output) + ".sha256").write_text(f"{sha256(args.state_output)}  {args.state_output.name}\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "body": body_candidate, "color": color_candidate,
                      "body_gain": body_gain, "color_unknown_reduction": color_unknown_reduction,
                      "body_track": body_track_final, "color_track": color_track_final, "gates": gates}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
