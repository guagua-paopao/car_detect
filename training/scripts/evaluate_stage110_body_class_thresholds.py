#!/usr/bin/env python3
"""Select exact-body per-class abstention thresholds on validation only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import evaluate_stage109_color_class_thresholds as thresholding
import evaluate_v2_decoupled_shared_validation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--body-specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-specialist-checkpoint-sha256", required=True)
    parser.add_argument("--body-specialist-subtype-threshold", type=float, default=0.50)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-color-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.935)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stratified(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "lighting",
        "vehicle_size",
        "weather",
        "occlusion_level",
        "source_dataset",
        "source_video",
    ):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if base.truthy(row.get("body_type_supervised")):
                groups[row.get(field) or "unknown"].append(index)
        result[field] = {}
        for group, indexes in sorted(groups.items()):
            result[field][group] = thresholding.metric(
                [outputs[index]["body_label"] for index in indexes],
                [outputs[index]["body_confidence"] for index in indexes],
                [rows[index]["body_type"] for index in indexes],
                thresholds,
            )
    return result


def fused(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    tracks: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if base.truthy(row.get("body_type_supervised")):
            tracks[row.get("track_key") or row.get("track_id") or f"row:{index}"].append(index)
    final_labels: list[str] = []
    final_truths: list[str] = []
    labels_by_track: dict[str, list[str]] = {}
    for track, indexes in tracks.items():
        observations = []
        for index in indexes:
            prediction = outputs[index]["body_label"]
            confidence = outputs[index]["body_confidence"]
            emitted = (
                prediction
                if prediction != "unknown" and confidence >= thresholds.get(prediction, 1.01)
                else "unknown"
            )
            observations.append((emitted, confidence, base.quality_weight(rows[index])))
        fused_labels = base.fuse_observations(observations)
        labels_by_track[track] = fused_labels
        final_labels.append(fused_labels[-1])
        final_truths.append(Counter(rows[index]["body_type"] for index in indexes).most_common(1)[0][0])
    selected = [index for index, label in enumerate(final_labels) if label != "unknown"]
    correct = sum(final_labels[index] == final_truths[index] for index in selected)
    total = len(final_truths)
    return {
        "track_final": {
            "evaluated": total,
            "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "coverage": len(selected) / total if total else 0.0,
            "unknown_rate": 1.0 - len(selected) / total if total else 1.0,
        },
        "stability": base.stability(labels_by_track),
        "window": 5,
        "minimum_share": 0.60,
        "minimum_margin": 0.15,
    }


def main() -> int:
    args = parse_args()
    for path, expected, role in (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256, "body checkpoint"),
        (
            args.body_specialist_checkpoint,
            args.expected_body_specialist_checkpoint_sha256,
            "body specialist checkpoint",
        ),
        (args.color_checkpoint, args.expected_color_checkpoint_sha256, "color checkpoint"),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "baseline checkpoint"),
    ):
        thresholding.assert_hash(path, expected, role)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = base.resolve_rows(args.manifest, args.datasets_safety_root)
    outputs = base.infer_pair(
        args,
        rows,
        args.body_checkpoint,
        args.color_checkpoint,
        expected_labels=labels,
        body_specialist_checkpoint_path=args.body_specialist_checkpoint,
    )
    baseline_outputs = base.infer_pair(
        args,
        rows,
        args.baseline_checkpoint,
        args.baseline_checkpoint,
        expected_labels=None,
    )
    indexes = [index for index, row in enumerate(rows) if base.truthy(row.get("body_type_supervised"))]
    if not indexes or any(not base.is_complex_row(rows[index]) for index in indexes):
        raise RuntimeError("expected a non-empty all-complex supervised body validation view")
    predictions = [outputs[index]["body_label"] for index in indexes]
    confidences = [outputs[index]["body_confidence"] for index in indexes]
    truths = [rows[index]["body_type"] for index in indexes]
    selection = thresholding.optimize_thresholds(
        predictions, confidences, truths, args.precision_target
    )
    thresholds = selection["thresholds"]
    candidate_static = thresholding.metric(predictions, confidences, truths, thresholds)
    candidate_track = fused(rows, outputs, thresholds)

    baseline_predictions = [baseline_outputs[index]["body_label"] for index in indexes]
    baseline_confidences = [baseline_outputs[index]["body_confidence"] for index in indexes]
    baseline_selection = base.select_threshold(
        baseline_predictions,
        baseline_confidences,
        truths,
        [value / 1000 for value in range(500, 991)],
        precision_gate=0.93,
    )
    baseline_threshold = float(baseline_selection["threshold"])
    baseline_static = base.metric_at_threshold(
        baseline_predictions, baseline_confidences, truths, baseline_threshold
    )
    baseline_track = base.fused_metrics(rows, baseline_outputs, "body", baseline_threshold)
    complex_coverage_gain = candidate_static["coverage"] - baseline_static["coverage"]
    gates = {
        "body_static_precision": candidate_static["precision"] >= 0.93,
        "body_static_coverage": candidate_static["coverage"] >= 0.45,
        "body_complex_coverage_gain": complex_coverage_gain >= 0.15,
        "body_track_precision": candidate_track["track_final"]["precision"] >= 0.93,
        "body_track_coverage": candidate_track["track_final"]["coverage"] >= 0.45,
        "body_track_stability": candidate_track["stability"]["transition_stability"] >= 0.95,
    }
    report = {
        "schema_version": "stage110-body-class-threshold-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": base.sha256(args.manifest),
            "labels_sha256": base.sha256(args.labels),
            "body_checkpoint_sha256": base.sha256(args.body_checkpoint),
            "body_specialist_checkpoint_sha256": base.sha256(args.body_specialist_checkpoint),
            "body_specialist_subtype_threshold": args.body_specialist_subtype_threshold,
            "color_checkpoint_sha256": base.sha256(args.color_checkpoint),
            "baseline_checkpoint_sha256": base.sha256(args.baseline_checkpoint),
        },
        "selection": selection,
        "static": {"candidate": candidate_static, "production_baseline": baseline_static},
        "comparison": {"body_complex_static_coverage_gain": complex_coverage_gain},
        "per_class": thresholding.per_class_metrics(
            predictions, confidences, truths, thresholds
        ),
        "stratified": stratified(rows, outputs, thresholds),
        "track_fusion": {"candidate": candidate_track, "production_baseline": baseline_track},
        "gates": {"gates": gates, "all_pass": all(gates.values())},
        "decision": (
            "body_component_validation_pass_pending_integrated_gate"
            if all(gates.values())
            else "body_component_rejected_fail_closed"
        ),
        "policy": {
            "split": "validation",
            "test_accessed": False,
            "frozen_video_used": False,
            "labels_rewritten": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        f"{base.sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "thresholds": thresholds,
                "static": candidate_static,
                "coverage_gain": complex_coverage_gain,
                "track": candidate_track,
                "gates": gates,
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
