#!/usr/bin/env python3
"""Select fail-closed per-predicted-color thresholds on validation only.

The shared CCTV validation taxonomy intentionally merges gray/silver and the
two fine warm-color pairs.  This script changes only the abstention policy: it
never relabels a prediction and never opens a test or frozen-video payload.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import evaluate_v2_decoupled_shared_validation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--body-specialist-checkpoint", type=Path)
    parser.add_argument("--expected-body-specialist-checkpoint-sha256")
    parser.add_argument("--body-specialist-subtype-threshold", type=float, default=0.65)
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


def emitted_labels(
    predictions: list[str], confidences: list[float], thresholds: dict[str, float]
) -> list[str]:
    return [
        prediction
        if prediction != "unknown" and confidence >= thresholds.get(prediction, 1.01)
        else "unknown"
        for prediction, confidence in zip(predictions, confidences)
    ]


def metric(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not (len(predictions) == len(confidences) == len(truths)):
        raise ValueError("metric inputs have different lengths")
    emitted = emitted_labels(predictions, confidences, thresholds)
    selected = [index for index, label in enumerate(emitted) if label != "unknown"]
    correct = sum(emitted[index] == truths[index] for index in selected)
    total = len(truths)
    return {
        "evaluated": total,
        "selected": len(selected),
        "correct": correct,
        "precision": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / total if total else 0.0,
        "unknown_rate": 1.0 - len(selected) / total if total else 1.0,
    }


def threshold_options(
    prediction: str,
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
) -> list[tuple[int, int, float]]:
    options: dict[tuple[int, int], float] = {(0, 0): 1.01}
    for step in range(500, 991):
        threshold = step / 1000.0
        indexes = [
            index
            for index, (label, confidence) in enumerate(zip(predictions, confidences))
            if label == prediction and confidence >= threshold
        ]
        selected = len(indexes)
        correct = sum(truths[index] == prediction for index in indexes)
        # Identical emission sets retain the stricter threshold.
        options[(selected, correct)] = max(threshold, options.get((selected, correct), 0.0))
    return [(selected, correct, threshold) for (selected, correct), threshold in options.items()]


def optimize_thresholds(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    precision_target: float,
) -> dict[str, Any]:
    if not 0.0 < precision_target <= 1.0:
        raise ValueError("precision target must be in (0, 1]")
    labels = sorted({label for label in predictions if label != "unknown"})
    # For each selected count, retain only the state with most correct samples.
    # This is an exact dynamic program because class-specific emission sets are
    # disjoint and both the constraint and objective are additive.
    states: dict[int, tuple[int, dict[str, float]]] = {0: (0, {})}
    option_counts: dict[str, int] = {}
    for label in labels:
        options = threshold_options(label, predictions, confidences, truths)
        option_counts[label] = len(options)
        next_states: dict[int, tuple[int, dict[str, float]]] = {}
        for prior_selected, (prior_correct, prior_path) in states.items():
            for selected, correct, threshold in options:
                total_selected = prior_selected + selected
                total_correct = prior_correct + correct
                incumbent = next_states.get(total_selected)
                path = {**prior_path, label: threshold}
                if incumbent is None or total_correct > incumbent[0]:
                    next_states[total_selected] = (total_correct, path)
                elif total_correct == incumbent[0]:
                    # Prefer the more conservative equivalent policy.
                    if sum(path.values()) > sum(incumbent[1].values()):
                        next_states[total_selected] = (total_correct, path)
        states = next_states
    passing = [
        (selected, correct, path)
        for selected, (correct, path) in states.items()
        if selected > 0 and correct / selected >= precision_target
    ]
    if not passing:
        raise RuntimeError("no class-threshold policy satisfies the precision target")
    selected, correct, thresholds = max(
        passing,
        key=lambda item: (item[0], item[1] / item[0], sum(item[2].values())),
    )
    return {
        "thresholds": thresholds,
        "precision_target": precision_target,
        "selected": selected,
        "correct": correct,
        "precision": correct / selected,
        "coverage": selected / len(truths) if truths else 0.0,
        "option_counts": option_counts,
        "algorithm": "exact additive dynamic program on validation-only 0.001 threshold grid",
    }


def per_class_metrics(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    emitted = emitted_labels(predictions, confidences, thresholds)
    result: dict[str, Any] = {}
    for label in sorted(set(truths) | {value for value in emitted if value != "unknown"}):
        if label == "unknown":
            continue
        truth_indexes = [index for index, truth in enumerate(truths) if truth == label]
        predicted_indexes = [index for index, value in enumerate(emitted) if value == label]
        selected_truth = [index for index in truth_indexes if emitted[index] != "unknown"]
        true_positives = sum(emitted[index] == label for index in truth_indexes)
        result[label] = {
            "support": len(truth_indexes),
            "predicted": len(predicted_indexes),
            "true_positives": true_positives,
            "precision": true_positives / len(predicted_indexes) if predicted_indexes else 0.0,
            "recall": true_positives / len(truth_indexes) if truth_indexes else 0.0,
            "coverage": len(selected_truth) / len(truth_indexes) if truth_indexes else 0.0,
            "unknown_rate": 1.0 - len(selected_truth) / len(truth_indexes) if truth_indexes else 1.0,
        }
    return result


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
            if base.truthy(row.get("color_supervised")):
                groups[row.get(field) or "unknown"].append(index)
        result[field] = {}
        for group, indexes in sorted(groups.items()):
            result[field][group] = metric(
                [outputs[index]["color_label"] for index in indexes],
                [outputs[index]["color_confidence"] for index in indexes],
                [rows[index]["color"] for index in indexes],
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
        if base.truthy(row.get("color_supervised")):
            tracks[row.get("track_key") or row.get("track_id") or f"row:{index}"].append(index)
    all_emitted: list[str] = []
    all_truths: list[str] = []
    labels_by_track: dict[str, list[str]] = {}
    for track, indexes in tracks.items():
        observations = []
        for index in indexes:
            prediction = outputs[index]["color_label"]
            confidence = outputs[index]["color_confidence"]
            emitted = (
                prediction
                if prediction != "unknown" and confidence >= thresholds.get(prediction, 1.01)
                else "unknown"
            )
            observations.append((emitted, confidence, base.quality_weight(rows[index])))
        fused_labels = base.fuse_observations(observations)
        labels_by_track[track] = fused_labels
        known_truths = [rows[index]["color"] for index in indexes]
        truth = Counter(known_truths).most_common(1)[0][0]
        all_emitted.append(fused_labels[-1])
        all_truths.append(truth)
    selected = [index for index, label in enumerate(all_emitted) if label != "unknown"]
    correct = sum(all_emitted[index] == all_truths[index] for index in selected)
    total = len(all_truths)
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


def assert_hash(path: Path, expected: str, role: str) -> None:
    actual = base.sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")


def main() -> int:
    args = parse_args()
    assert_hash(args.manifest, args.expected_manifest_sha256, "manifest")
    assert_hash(args.labels, args.expected_labels_sha256, "labels")
    assert_hash(args.body_checkpoint, args.expected_body_checkpoint_sha256, "body checkpoint")
    assert_hash(args.color_checkpoint, args.expected_color_checkpoint_sha256, "color checkpoint")
    assert_hash(args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "baseline checkpoint")
    if bool(args.body_specialist_checkpoint) != bool(args.expected_body_specialist_checkpoint_sha256):
        raise RuntimeError("specialist path and expected SHA256 must be provided together")
    if args.body_specialist_checkpoint:
        assert_hash(
            args.body_specialist_checkpoint,
            args.expected_body_specialist_checkpoint_sha256,
            "body specialist checkpoint",
        )
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
    indexes = [index for index, row in enumerate(rows) if base.truthy(row.get("color_supervised"))]
    if not indexes or any(not base.is_complex_row(rows[index]) for index in indexes):
        raise RuntimeError("expected a non-empty all-complex supervised color validation view")
    predictions = [outputs[index]["color_label"] for index in indexes]
    confidences = [outputs[index]["color_confidence"] for index in indexes]
    truths = [rows[index]["color"] for index in indexes]
    selection = optimize_thresholds(predictions, confidences, truths, args.precision_target)
    thresholds = selection["thresholds"]
    candidate_static = metric(predictions, confidences, truths, thresholds)
    candidate_track = fused(rows, outputs, thresholds)

    baseline_predictions = [baseline_outputs[index]["color_label"] for index in indexes]
    baseline_confidences = [baseline_outputs[index]["color_confidence"] for index in indexes]
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
    baseline_track = base.fused_metrics(rows, baseline_outputs, "color", baseline_threshold)
    unknown_reduction = (
        (baseline_static["unknown_rate"] - candidate_static["unknown_rate"])
        / baseline_static["unknown_rate"]
        if baseline_static["unknown_rate"] > 0
        else 0.0
    )
    gates = {
        "color_static_precision": candidate_static["precision"] >= 0.93,
        "color_static_coverage": candidate_static["coverage"] >= 0.25,
        "color_complex_unknown_reduction": unknown_reduction >= 0.20,
        "color_track_precision": candidate_track["track_final"]["precision"] >= 0.93,
        "color_track_coverage": candidate_track["track_final"]["coverage"] >= 0.25,
        "color_track_stability": candidate_track["stability"]["transition_stability"] >= 0.95,
    }
    report = {
        "schema_version": "stage109-color-class-threshold-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": base.sha256(args.manifest),
            "labels_sha256": base.sha256(args.labels),
            "body_checkpoint_sha256": base.sha256(args.body_checkpoint),
            "body_specialist_checkpoint_sha256": (
                base.sha256(args.body_specialist_checkpoint)
                if args.body_specialist_checkpoint
                else None
            ),
            "color_checkpoint_sha256": base.sha256(args.color_checkpoint),
            "baseline_checkpoint_sha256": base.sha256(args.baseline_checkpoint),
        },
        "selection": selection,
        "static": {"candidate": candidate_static, "production_baseline": baseline_static},
        "comparison": {
            "color_complex_static_unknown_relative_reduction": unknown_reduction,
        },
        "per_class": per_class_metrics(predictions, confidences, truths, thresholds),
        "stratified": stratified(rows, outputs, thresholds),
        "track_fusion": {"candidate": candidate_track, "production_baseline": baseline_track},
        "gates": {"gates": gates, "all_pass": all(gates.values())},
        "decision": (
            "color_component_validation_pass_pending_integrated_gate"
            if all(gates.values())
            else "color_component_rejected_fail_closed"
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
                "unknown_reduction": unknown_reduction,
                "track": candidate_track,
                "gates": gates,
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
