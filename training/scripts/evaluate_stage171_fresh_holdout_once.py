#!/usr/bin/env python3
"""Evaluate the Stage169 pair on the sealed Stage170 holdout exactly once.

All candidate thresholds and track policies come from validation evidence.
The test set is never used for selection, threshold search, calibration, or
temperature fitting.  A durable claim is written before the manifest is read;
after that point a failed execution is still considered consumed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage109_color_class_thresholds as thresholding  # noqa: E402
import evaluate_stage159_component_validation as component  # noqa: E402
import evaluate_stage166_selective_short_tracks as selective_tracks  # noqa: E402
import evaluate_v2_decoupled_shared_validation as validation  # noqa: E402


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen_video")
TRACK_KEYS = (
    "stage170_group",
    "track_key",
    "track_id",
    "track_group",
    "stage159_validation_group",
    "stage157_group",
)
ORDER_KEYS = ("frame_index", "frame_id", "timestamp", "source_frame_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage170-state", type=Path, required=True)
    parser.add_argument("--expected-stage170-state-sha256", required=True)
    parser.add_argument("--stage169-state", type=Path, required=True)
    parser.add_argument("--expected-stage169-state-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--expected-production-config-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {role} SHA256 mismatch: {path}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def valid_thresholds(value: Any, role: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Stage169 {role} thresholds are missing")
    result = {str(label): float(threshold) for label, threshold in value.items()}
    if any(not 0.0 < threshold <= 1.01 for threshold in result.values()):
        raise RuntimeError(f"Stage169 {role} thresholds are invalid")
    return result


def load_stage170(path: Path, expected_sha256: str) -> tuple[dict[str, Any], Path]:
    require_sha(path, expected_sha256, "Stage170 state")
    state = read_json(path)
    if state.get("status") != "fresh_holdout_sealed_pending_one_time_inference":
        raise RuntimeError("Stage170 fresh holdout is not sealed for one-time inference")
    authorization = state.get("authorization", {})
    if authorization.get("one_time_independent_test_inference") is not True:
        raise RuntimeError("Stage170 did not authorize one-time inference")
    for key in ("onnx_backend_gates", "frozen_video_replay", "deployment"):
        if authorization.get(key) is not False:
            raise RuntimeError(f"unsafe Stage170 authorization: {key}")
    for key in (
        "test_inference_run",
        "test_used_for_selection",
        "threshold_or_temperature_search",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if state.get(key) is not False:
            raise RuntimeError(f"unsafe or consumed Stage170 state: {key}")
    report_path = Path(str(state.get("report", "")))
    require_sha(report_path, str(state.get("report_sha256", "")), "Stage170 build report")
    report = read_json(report_path)
    if report.get("status") != "pass_fresh_holdout_built_and_sealed":
        raise RuntimeError("Stage170 build report is not passing")
    manifest = Path(str(state.get("manifest", "")))
    require_sha(manifest, str(state.get("manifest_sha256", "")), "Stage170 manifest")
    if report.get("output_manifest_sha256") != state.get("manifest_sha256"):
        raise RuntimeError("Stage170 report/state manifest pin mismatch")
    return state, manifest


def load_stage169(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha(path, expected_sha256, "Stage169 state")
    state = read_json(path)
    if state.get("status") != "integrated_candidate_pass_stage160_holdout_build_authorized":
        raise RuntimeError("Stage169 integrated candidate is not validation-qualified")
    for key in (
        "test_accessed",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if state.get(key) is not False:
            raise RuntimeError(f"unsafe Stage169 state: {key}")
    body = state.get("body")
    color = state.get("color")
    if not isinstance(body, dict) or not isinstance(color, dict):
        raise RuntimeError("Stage169 component evidence is incomplete")
    parameters = {
        "body_thresholds": valid_thresholds(body.get("thresholds"), "body"),
        "color_thresholds": valid_thresholds(color.get("thresholds"), "color"),
        "body_track_config": body.get("track", {}).get("config"),
        "body_short_thresholds": valid_thresholds(
            body.get("track", {}).get("short_thresholds"), "body short-track"
        ),
        "color_track_policy": {
            "window": int(color.get("track", {}).get("window", 0)),
            "minimum_share": float(color.get("track", {}).get("minimum_share", -1.0)),
            "minimum_margin": float(color.get("track", {}).get("minimum_margin", -1.0)),
        },
    }
    body_track = parameters["body_track_config"]
    if not isinstance(body_track, dict) or set(body_track) != {
        "short_track_max_length",
        "short_precision_target",
        "short_minimum_quality",
        "short_unknown_hold",
    }:
        raise RuntimeError("Stage169 fixed body track policy is invalid")
    if parameters["color_track_policy"] != {
        "window": 5,
        "minimum_share": 0.6,
        "minimum_margin": 0.15,
    }:
        raise RuntimeError("Stage169 fixed color track policy changed")
    for role, evidence in (("body", body), ("color", color)):
        checkpoint = Path(str(evidence.get("checkpoint", "")))
        require_sha(checkpoint, str(evidence.get("checkpoint_sha256", "")), f"Stage169 {role} checkpoint")
    return state, parameters


def validate_cross_stage(stage170: dict[str, Any], stage169_path: Path, stage169_sha: str) -> None:
    report = read_json(Path(str(stage170["report"])))
    if Path(str(report.get("authorization_state", ""))).resolve() != stage169_path.resolve():
        raise RuntimeError("Stage170 was not built from this Stage169 state")
    if str(report.get("authorization_state_sha256", "")).lower() != stage169_sha.lower():
        raise RuntimeError("Stage170/Stage169 authorization SHA256 mismatch")


def resolve_test_rows(manifest: Path, safety_root: Path) -> list[dict[str, str]]:
    safety_root = safety_root.resolve()
    if not safety_root.is_dir():
        raise RuntimeError("datasets safety root is missing")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("fresh test manifest has no rows")
    for line, row in enumerate(rows, start=2):
        if row.get("split") != "test":
            raise RuntimeError(f"fresh holdout contains split={row.get('split')} at line {line}")
        if row.get("review_status") != "approved":
            raise RuntimeError(f"unapproved fresh holdout row {line}")
        searchable = " ".join(str(value).lower() for value in row.values())
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"frozen marker in fresh holdout row {line}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"fresh holdout image escapes safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["_resolved_image_path"] = str(resolved)
    return rows


def head_indexes(rows: list[dict[str, str]], head: str) -> list[int]:
    field = "body_type_supervised" if head == "body" else "color_supervised"
    return [index for index, row in enumerate(rows) if truthy(row.get(field))]


def is_complex(row: dict[str, str]) -> bool:
    return validation.is_complex_row(row)


def track_name(row: dict[str, str], index: int) -> str:
    return next((str(row[key]) for key in TRACK_KEYS if row.get(key)), f"row:{index}")


def order_key(row: dict[str, str], index: int) -> tuple[int, int, Any, int]:
    for position, key in enumerate(ORDER_KEYS):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return position, 0, float(value), index
            except ValueError:
                return position, 1, str(value), index
    return len(ORDER_KEYS), 0, index, index


def grouped_indexes(rows: list[dict[str, str]], head: str) -> dict[str, list[int]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index in head_indexes(rows, head):
        groups[track_name(rows[index], index)].append(index)
    return {
        group: sorted(indexes, key=lambda index: order_key(rows[index], index))
        for group, indexes in groups.items()
    }


def single_track_truth(
    rows: list[dict[str, str]], indexes: list[int], truth_key: str,
    transform: Callable[[str], str] = lambda value: value,
) -> str:
    truths = {transform(rows[index][truth_key]) for index in indexes}
    if len(truths) != 1:
        raise RuntimeError(f"inconsistent {truth_key} truth inside Stage170 vehicle group")
    return next(iter(truths))


def fixed_fusion_track(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    thresholds: dict[str, float],
    truth_transform: Callable[[str], str] = lambda value: value,
) -> dict[str, Any]:
    groups = grouped_indexes(rows, head)
    labels_by_track: dict[str, list[str]] = {}
    final_truths: list[str] = []
    for group, indexes in groups.items():
        observations = []
        for index in indexes:
            label = outputs[index][f"{head}_label"]
            confidence = float(outputs[index][f"{head}_confidence"])
            emitted = label if label != "unknown" and confidence >= thresholds.get(label, 1.01) else "unknown"
            observations.append((emitted, confidence, validation.quality_weight(rows[index])))
        labels_by_track[group] = validation.fuse_observations(observations)
        final_truths.append(single_track_truth(
            rows, indexes, "body_type" if head == "body" else "color", truth_transform
        ))
    final_labels = [labels_by_track[group][-1] for group in groups]
    selected = [index for index, label in enumerate(final_labels) if label != "unknown"]
    correct = sum(final_labels[index] == final_truths[index] for index in selected)
    return {
        "track_final": {
            "evaluated": len(final_truths),
            "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "coverage": len(selected) / len(final_truths) if final_truths else 0.0,
            "unknown_rate": 1.0 - len(selected) / len(final_truths) if final_truths else 1.0,
        },
        "stability": validation.stability(labels_by_track),
        "window": 5,
        "minimum_share": 0.6,
        "minimum_margin": 0.15,
    }


def fixed_body_track(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    base_thresholds: dict[str, float],
    short_thresholds: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    tracks = grouped_indexes(rows, "body")
    raw = {
        group: [
            (
                outputs[index]["body_label"],
                float(outputs[index]["body_confidence"]),
                validation.quality_weight(rows[index]),
            )
            for index in indexes
        ]
        for group, indexes in tracks.items()
    }
    truths = {
        group: single_track_truth(rows, indexes, "body_type")
        for group, indexes in tracks.items()
    }
    result = selective_tracks.metric_for_policy(
        tracks, raw, truths, base_thresholds, short_thresholds, config
    )
    result["selection_search"] = False
    result["policy_source"] = "Stage169 validation-pinned Stage168/Stage166 policy"
    return result


def static_metric(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    indexes: list[int],
    head: str,
    thresholds: dict[str, float],
    truth_transform: Callable[[str], str],
) -> dict[str, Any]:
    truth_key = "body_type" if head == "body" else "color"
    return thresholding.metric(
        [outputs[index][f"{head}_label"] for index in indexes],
        [outputs[index][f"{head}_confidence"] for index in indexes],
        [truth_transform(rows[index][truth_key]) for index in indexes],
        thresholds,
    )


def metric_bundle(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    thresholds: dict[str, float],
    *,
    truth_transform: Callable[[str], str] = lambda value: value,
    track: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexes = head_indexes(rows, head)
    complex_indexes = [index for index in indexes if is_complex(rows[index])]
    truth_key = "body_type" if head == "body" else "color"
    predictions = [outputs[index][f"{head}_label"] for index in indexes]
    confidences = [outputs[index][f"{head}_confidence"] for index in indexes]
    truths = [truth_transform(rows[index][truth_key]) for index in indexes]
    stratified: dict[str, Any] = {}
    for field in ("lighting", "vehicle_size", "weather", "occlusion_level", "source_dataset", "camera_id"):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index in indexes:
            groups[rows[index].get(field) or "unknown"].append(index)
        stratified[field] = {
            group: static_metric(rows, outputs, group_indexes, head, thresholds, truth_transform)
            for group, group_indexes in sorted(groups.items())
        }
    return {
        "supervised_rows": len(indexes),
        "complex_supervised_rows": len(complex_indexes),
        "static": thresholding.metric(predictions, confidences, truths, thresholds),
        "complex_static": static_metric(
            rows, outputs, complex_indexes, head, thresholds, truth_transform
        ),
        "per_class": thresholding.per_class_metrics(predictions, confidences, truths, thresholds),
        "stratified": stratified,
        "track": track or fixed_fusion_track(rows, outputs, head, thresholds, truth_transform),
    }


def production_truth(head: str, value: str) -> str:
    return component.production_truth_label(head, value)


def relative_unknown_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def post_test_authorization(passed: bool) -> dict[str, bool]:
    return {
        "onnx_backend_gates": passed,
        "frozen_video_replay": False,
        "deployment": False,
    }


def make_claim(args: argparse.Namespace, manifest: Path) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=False)
    claim = args.output_root / "ONE_TIME_TEST_CLAIM.json"
    write_json_atomic(claim, {
        "schema_version": "stage171-one-time-test-claim-v1",
        "claimed_at": now(),
        "stage170_state": str(args.stage170_state.resolve()),
        "stage170_state_sha256": sha256(args.stage170_state),
        "stage169_state": str(args.stage169_state.resolve()),
        "stage169_state_sha256": sha256(args.stage169_state),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "test_access_after_claim_only": True,
        "retry_permitted": False,
        "frozen_video_used": False,
    })
    return claim


def persist_result(output_root: Path, claim: Path, report: dict[str, Any]) -> None:
    report_path = output_root / "stage171-fresh-holdout-test-report.json"
    write_json_atomic(report_path, report)
    state = {
        "schema_version": "stage171-fresh-holdout-test-state-v1",
        "created_at": report["created_at"],
        "status": report["status"],
        "decision": report.get("decision", "execution_failed_fail_closed"),
        "claim": str(claim.resolve()),
        "claim_sha256": sha256(claim),
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
        "authorization": report["authorization"],
        "test_accessed": True,
        "test_used_for_selection": False,
        "threshold_or_temperature_search": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    state_path = output_root / "stage171.state.json"
    write_json_atomic(state_path, state)
    sums = output_root / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (claim, report_path, state_path)
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite one-time evidence: {args.output_root}")
    pinned = (
        (args.stage170_state, args.expected_stage170_state_sha256, "Stage170 state"),
        (args.stage169_state, args.expected_stage169_state_sha256, "Stage169 state"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.production_config, args.expected_production_config_sha256, "production configuration"),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "production checkpoint"),
    )
    for path, expected, role in pinned:
        require_sha(path, expected, role)
    stage170, manifest = load_stage170(args.stage170_state, args.expected_stage170_state_sha256)
    stage169, parameters = load_stage169(args.stage169_state, args.expected_stage169_state_sha256)
    validate_cross_stage(stage170, args.stage169_state, args.expected_stage169_state_sha256)
    if str(stage169.get("inputs", {}).get("production_config_sha256", "")).lower() != sha256(args.production_config):
        raise RuntimeError("Stage169/Stage171 production configuration pin mismatch")
    labels = read_json(args.labels)
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 labels are required")
    body_path = Path(stage169["body"]["checkpoint"])
    color_path = Path(stage169["color"]["checkpoint"])
    claim = make_claim(args, manifest)
    try:
        rows = resolve_test_rows(manifest, args.datasets_safety_root)
        candidate_pair = validation.infer_pair(
            args, rows, body_path, color_path, expected_labels=labels,
            body_specialist_checkpoint_path=None,
        )
        candidate_color_only = component.infer_color_fine(args, rows, color_path, labels)
        candidate = [
            {
                "body_label": paired["body_label"],
                "body_confidence": paired["body_confidence"],
                "color_label": colored["color_label"],
                "color_confidence": colored["color_confidence"],
            }
            for paired, colored in zip(candidate_pair, candidate_color_only)
        ]
        baseline = validation.infer_pair(
            args, rows, args.baseline_checkpoint, args.baseline_checkpoint,
            expected_labels=None, body_specialist_checkpoint_path=None,
        )
        candidate_body_track = fixed_body_track(
            rows,
            candidate,
            parameters["body_thresholds"],
            parameters["body_short_thresholds"],
            parameters["body_track_config"],
        )
        candidate_body = metric_bundle(
            rows, candidate, "body", parameters["body_thresholds"], track=candidate_body_track
        )
        candidate_color = metric_bundle(rows, candidate, "color", parameters["color_thresholds"])
        production = read_json(args.production_config)
        analytics = production.get("vehicle_analytics", production)
        production_body_threshold = float(analytics.get("type_threshold", 0.0))
        production_color_threshold = float(analytics.get("color_threshold", 0.0))
        if not 0.0 < production_body_threshold <= 1.0 or not 0.0 < production_color_threshold <= 1.0:
            raise RuntimeError("production thresholds are missing or invalid")
        baseline_body_thresholds = {
            output["body_label"]: production_body_threshold
            for output in baseline if output["body_label"] != "unknown"
        }
        baseline_color_thresholds = {
            output["color_label"]: production_color_threshold
            for output in baseline if output["color_label"] != "unknown"
        }
        baseline_body = metric_bundle(rows, baseline, "body", baseline_body_thresholds)
        baseline_color = metric_bundle(
            rows, baseline, "color", baseline_color_thresholds,
            truth_transform=lambda value: production_truth("color", value),
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
        for path, expected, role in pinned:
            require_sha(path, expected, role)
        require_sha(manifest, str(stage170["manifest_sha256"]), "Stage170 manifest after inference")
        require_sha(body_path, str(stage169["body"]["checkpoint_sha256"]), "body checkpoint after inference")
        require_sha(color_path, str(stage169["color"]["checkpoint_sha256"]), "color checkpoint after inference")
        passed = all(gates.values())
        report = {
            "schema_version": "stage171-fresh-holdout-test-once-v1",
            "created_at": now(),
            "status": "complete_one_time_independent_test",
            "decision": "independent_test_gates_pass" if passed else "independent_test_gates_fail_closed",
            "authorization": post_test_authorization(passed),
            "inputs": {
                "stage170_state_sha256": sha256(args.stage170_state),
                "stage169_state_sha256": sha256(args.stage169_state),
                "manifest_sha256": sha256(manifest),
                "labels_sha256": sha256(args.labels),
                "body_checkpoint_sha256": sha256(body_path),
                "color_checkpoint_sha256": sha256(color_path),
                "production_checkpoint_sha256": sha256(args.baseline_checkpoint),
                "production_config_sha256": sha256(args.production_config),
            },
            "fixed_validation_parameters": {
                **parameters,
                "production_body_threshold": production_body_threshold,
                "production_color_threshold": production_color_threshold,
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
            "all_gates_pass": passed,
            "policy": {
                "split": "fresh_test",
                "test_accessed": True,
                "test_used_for_selection": False,
                "threshold_or_temperature_search": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
            },
        }
        persist_result(args.output_root, claim, report)
        print(json.dumps({"decision": report["decision"], "gates": gates}, ensure_ascii=False))
        return 0 if passed else 2
    except Exception as error:
        report = {
            "schema_version": "stage171-fresh-holdout-test-once-v1",
            "created_at": now(),
            "status": "one_time_test_consumed_execution_failed_fail_closed",
            "decision": "execution_failed_fail_closed",
            "authorization": post_test_authorization(False),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "policy": {
                "test_accessed": True,
                "test_used_for_selection": False,
                "threshold_or_temperature_search": False,
                "retry_permitted": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
            },
        }
        persist_result(args.output_root, claim, report)
        print(json.dumps({"decision": report["decision"], "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
