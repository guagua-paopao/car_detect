#!/usr/bin/env python3
"""Validate selective singleton/short-track admission without lowering thresholds."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import evaluate_stage165_track_frontier as stage165

component = stage165.component


LONG_TRACK_CONFIG = {
    "window": 5,
    "minimum_observations": 2,
    "entry_share": 0.70,
    "entry_margin": 0.0,
    "switch_share": 0.70,
    "switch_margin": 0.15,
    "switch_patience": 3,
    "unknown_hold": 2,
}

SHORT_TRACK_CONFIG = {
    "window": 3,
    "minimum_observations": 1,
    "entry_share": 0.60,
    "entry_margin": 0.15,
    "switch_share": 0.80,
    "switch_margin": 0.25,
    "switch_patience": 2,
}


def no_lower_thresholds(
    base_thresholds: dict[str, float],
    learned_thresholds: dict[str, float],
) -> dict[str, float]:
    labels = set(base_thresholds) | set(learned_thresholds)
    return {
        label: max(base_thresholds.get(label, 1.01), learned_thresholds.get(label, 1.01))
        for label in labels
    }


def short_track_thresholds(rows, outputs, head, indexes, base_thresholds, precision_target):
    predictions = [outputs[index][component.prediction_key(head)] for index in indexes]
    confidences = [outputs[index][component.confidence_key(head)] for index in indexes]
    truths = [rows[index][component.label_key(head)] for index in indexes]
    learned = component.thresholding.optimize_thresholds(
        predictions, confidences, truths, precision_target
    )["thresholds"]
    return no_lower_thresholds(base_thresholds, learned)


def summarize_track_metrics(labels_by_track, truths_by_track, lengths_by_track):
    def score(track_names):
        final_labels = [labels_by_track[name][-1] for name in track_names]
        final_truths = [truths_by_track[name] for name in track_names]
        selected = [index for index, label in enumerate(final_labels) if label != "unknown"]
        correct = sum(final_labels[index] == final_truths[index] for index in selected)
        return {
            "evaluated": len(track_names),
            "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "coverage": len(selected) / len(track_names) if track_names else 0.0,
            "unknown_rate": 1.0 - len(selected) / len(track_names) if track_names else 1.0,
        }

    names = list(labels_by_track)
    return {
        "track_final": score(names),
        "stability": component.base.stability(labels_by_track),
        "by_length": {
            "1": score([name for name in names if lengths_by_track[name] == 1]),
            "2": score([name for name in names if lengths_by_track[name] == 2]),
            "3_plus": score([name for name in names if lengths_by_track[name] >= 3]),
        },
    }


def metric_for_policy(
    tracks,
    raw_observations,
    truths_by_track,
    base_thresholds,
    short_thresholds,
    config: dict[str, Any],
):
    labels_by_track = {}
    lengths_by_track = {track: len(indexes) for track, indexes in tracks.items()}
    for track in tracks:
        length = lengths_by_track[track]
        observations = []
        if length <= config["short_track_max_length"]:
            for label, confidence, quality in raw_observations[track]:
                emitted = (
                    label
                    if label != "unknown"
                    and confidence >= short_thresholds.get(label, 1.01)
                    and quality >= config["short_minimum_quality"]
                    else "unknown"
                )
                observations.append((emitted, confidence, quality))
            labels_by_track[track] = stage165.frontier_fuse(
                observations,
                **SHORT_TRACK_CONFIG,
                unknown_hold=config["short_unknown_hold"],
            )
        else:
            for label, confidence, quality in raw_observations[track]:
                emitted = label if label != "unknown" and confidence >= base_thresholds.get(label, 1.01) else "unknown"
                observations.append((emitted, confidence, quality))
            labels_by_track[track] = stage165.frontier_fuse(observations, **LONG_TRACK_CONFIG)

    result = summarize_track_metrics(labels_by_track, truths_by_track, lengths_by_track)
    result["config"] = config
    result["short_thresholds"] = short_thresholds
    result["passes"] = (
        result["track_final"]["precision"] >= 0.93
        and result["track_final"]["coverage"] >= 0.45
        and result["stability"]["transition_stability"] >= 0.95
    )
    return result


def track_metrics(rows, outputs, head, thresholds):
    tracks: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not component.supervised(row, head):
            continue
        group = (
            row.get("track_key")
            or row.get("track_id")
            or row.get("track_group")
            or row.get("stage159_validation_group")
            or row.get("stage157_group")
            or f"row:{index}"
        )
        tracks[group].append(index)

    raw_observations = {}
    truths_by_track = {}
    for track, indexes in tracks.items():
        raw_observations[track] = [
            (
                outputs[index][component.prediction_key(head)],
                outputs[index][component.confidence_key(head)],
                component.base.quality_weight(rows[index]),
            )
            for index in indexes
        ]
        truths_by_track[track] = Counter(
            rows[index][component.label_key(head)] for index in indexes
        ).most_common(1)[0][0]

    candidates = []
    threshold_evidence = {}
    for short_track_max_length in (1, 2):
        indexes = [
            index
            for track, track_indexes in tracks.items()
            if len(track_indexes) <= short_track_max_length
            for index in track_indexes
        ]
        for short_precision_target in (0.935, 0.945, 0.955, 0.965, 0.975):
            short_threshold = short_track_thresholds(
                rows, outputs, head, indexes, thresholds, short_precision_target
            )
            threshold_evidence[f"lte{short_track_max_length}@{short_precision_target:.3f}"] = short_threshold
            for short_minimum_quality in (0.0, 0.20, 0.35, 0.50, 0.65, 0.75):
                for short_unknown_hold in (0, 1):
                    candidates.append(metric_for_policy(
                        tracks,
                        raw_observations,
                        truths_by_track,
                        thresholds,
                        short_threshold,
                        {
                            "short_track_max_length": short_track_max_length,
                            "short_precision_target": short_precision_target,
                            "short_minimum_quality": short_minimum_quality,
                            "short_unknown_hold": short_unknown_hold,
                        },
                    ))

    passing = [value for value in candidates if value["passes"]]
    precision_stable = [
        value for value in candidates
        if value["track_final"]["precision"] >= 0.93
        and value["stability"]["transition_stability"] >= 0.95
    ]
    selected = max(
        passing or precision_stable or candidates,
        key=lambda value: (
            value["track_final"]["coverage"],
            value["track_final"]["precision"],
            value["stability"]["transition_stability"],
        ),
    )
    selected = dict(selected)
    selected["selection_search"] = {
        "algorithm": "validation-only per-class short-track threshold plus crop-quality grid",
        "candidate_count": len(candidates),
        "passing_count": len(passing),
        "precision_stability_count": len(precision_stable),
        "selection": "maximum coverage among all track-gate-passing policies; otherwise maximum coverage preserving precision and stability",
        "threshold_evidence": threshold_evidence,
        "base_thresholds_never_lowered": all(
            learned.get(label, 1.01) >= value
            for learned in threshold_evidence.values()
            for label, value in thresholds.items()
        ),
        "test_accessed": False,
        "frozen_video_used": False,
    }
    return selected


def main() -> int:
    component.track_metrics = track_metrics
    return component.main()


if __name__ == "__main__":
    raise SystemExit(main())
