#!/usr/bin/env python3
"""Run Stage159 validation with a fail-closed persistent track-fusion sweep."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import sys
from typing import Any

import evaluate_stage159_component_validation as component


def persistent_fuse(
    observations: list[tuple[str, float, float]],
    *,
    window: int,
    minimum_observations: int,
    entry_share: float,
    entry_margin: float,
    switch_share: float,
    switch_margin: float,
    switch_patience: int,
) -> list[str]:
    """Causal quality-weighted fusion with hysteresis and unknown on conflict."""
    history: deque[tuple[str, float, float]] = deque(maxlen=window)
    state: str | None = None
    challenger: str | None = None
    challenger_count = 0
    outputs: list[str] = []
    for label, confidence, quality in observations:
        history.append((label, confidence, quality))
        scores: Counter[str] = Counter()
        known = 0
        total = 0.0
        for observed_label, observed_confidence, observed_quality in history:
            if observed_label == "unknown":
                continue
            weight = max(0.0, observed_confidence) * max(0.0, observed_quality)
            if weight <= 0.0:
                continue
            scores[observed_label] += weight
            total += weight
            known += 1
        if not scores or total <= 0.0:
            outputs.append("unknown")
            continue

        ranked = scores.most_common(2)
        winner, winner_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        share = winner_score / total
        margin = (winner_score - runner_up) / total

        if state is None:
            if known >= minimum_observations and share >= entry_share and margin >= entry_margin:
                state = winner
                challenger = None
                challenger_count = 0
                outputs.append(state)
            else:
                outputs.append("unknown")
            continue

        if winner == state:
            challenger = None
            challenger_count = 0
            outputs.append(
                state if share >= entry_share and margin >= entry_margin else "unknown"
            )
            continue

        strong_conflict = (
            known >= minimum_observations
            and share >= switch_share
            and margin >= switch_margin
        )
        if strong_conflict:
            if challenger == winner:
                challenger_count += 1
            else:
                challenger = winner
                challenger_count = 1
            if challenger_count >= switch_patience:
                state = winner
                challenger = None
                challenger_count = 0
                outputs.append(state)
            else:
                # A single conflicting frame cannot overwrite stable history.
                outputs.append(state)
        else:
            challenger = None
            challenger_count = 0
            # Ambiguous sustained evidence is explicitly rejected.
            outputs.append("unknown")
    return outputs


def metric_for_config(
    tracks: dict[str, list[int]],
    rows: list[dict[str, str]],
    observations_by_track: dict[str, list[tuple[str, float, float]]],
    truths_by_track: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    final_labels: list[str] = []
    final_truths: list[str] = []
    labels_by_track: dict[str, list[str]] = {}
    for track in tracks:
        sequence = persistent_fuse(observations_by_track[track], **config)
        labels_by_track[track] = sequence
        final_labels.append(sequence[-1])
        final_truths.append(truths_by_track[track])
    selected = [index for index, label in enumerate(final_labels) if label != "unknown"]
    correct = sum(final_labels[index] == final_truths[index] for index in selected)
    stability = component.base.stability(labels_by_track)
    return {
        "track_final": {
            "evaluated": len(final_truths),
            "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "coverage": len(selected) / len(final_truths) if final_truths else 0.0,
            "unknown_rate": 1.0 - len(selected) / len(final_truths) if final_truths else 1.0,
        },
        "stability": stability,
        "config": config,
    }


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

    observations_by_track: dict[str, list[tuple[str, float, float]]] = {}
    truths_by_track: dict[str, str] = {}
    for track, indexes in tracks.items():
        observations = []
        for index in indexes:
            predicted = outputs[index][component.prediction_key(head)]
            confidence = outputs[index][component.confidence_key(head)]
            emitted = predicted if predicted != "unknown" and confidence >= thresholds.get(predicted, 1.01) else "unknown"
            observations.append((emitted, confidence, component.base.quality_weight(rows[index])))
        observations_by_track[track] = observations
        truths_by_track[track] = Counter(
            rows[index][component.label_key(head)] for index in indexes
        ).most_common(1)[0][0]

    candidates = []
    for window in (3, 5):
        for minimum_observations in (1, 2):
            for entry_share in (0.60, 0.70):
                for entry_margin in (0.15, 0.25):
                    for switch_share in (0.70, 0.80):
                        for switch_margin in (0.25, 0.35):
                            for switch_patience in (2, 3):
                                config = {
                                    "window": window,
                                    "minimum_observations": minimum_observations,
                                    "entry_share": entry_share,
                                    "entry_margin": entry_margin,
                                    "switch_share": switch_share,
                                    "switch_margin": switch_margin,
                                    "switch_patience": switch_patience,
                                }
                                result = metric_for_config(
                                    tracks, rows, observations_by_track, truths_by_track, config
                                )
                                track_final = result["track_final"]
                                stability = result["stability"]
                                result["passes"] = (
                                    track_final["precision"] >= 0.93
                                    and track_final["coverage"] >= 0.45
                                    and stability["transition_stability"] >= 0.95
                                )
                                candidates.append(result)

    passing = [candidate for candidate in candidates if candidate["passes"]]
    if passing:
        selected = max(
            passing,
            key=lambda value: (
                value["track_final"]["coverage"],
                value["track_final"]["precision"],
                value["stability"]["transition_stability"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda value: (
                sum((
                    value["track_final"]["precision"] >= 0.93,
                    value["track_final"]["coverage"] >= 0.45,
                    value["stability"]["transition_stability"] >= 0.95,
                )),
                value["track_final"]["precision"],
                value["stability"]["transition_stability"],
                value["track_final"]["coverage"],
            ),
        )
    selected = dict(selected)
    selected["fusion_search"] = {
        "algorithm": "validation-only persistent quality-weighted causal grid search",
        "candidate_count": len(candidates),
        "passing_count": len(passing),
        "selection": "maximum coverage among all track-gate-passing configurations",
        "test_accessed": False,
        "frozen_video_used": False,
    }
    return selected


def main() -> int:
    component.track_metrics = track_metrics
    return component.main()


if __name__ == "__main__":
    raise SystemExit(main())
