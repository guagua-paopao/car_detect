#!/usr/bin/env python3
"""Run validation-only body evaluation with an explicit track P/C frontier."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

import evaluate_stage159_component_validation as component


def frontier_fuse(
    observations: list[tuple[str, float, float]],
    *,
    window: int,
    minimum_observations: int,
    entry_share: float,
    entry_margin: float,
    switch_share: float,
    switch_margin: float,
    switch_patience: int,
    unknown_hold: int,
) -> list[str]:
    """Causal fusion with singleton admission and bounded unknown holding.

    Inputs have already passed the validation-selected class threshold. A lone
    known observation can therefore enter when ``minimum_observations`` is one.
    Unknown frames can retain a stable state only for ``unknown_hold`` frames;
    ambiguous known evidence is never hidden by that hold.
    """
    history: deque[tuple[str, float, float]] = deque(maxlen=window)
    state: str | None = None
    challenger: str | None = None
    challenger_count = 0
    unknown_streak = 0
    outputs: list[str] = []

    for label, confidence, quality in observations:
        history.append((label, confidence, quality))
        if label == "unknown":
            unknown_streak += 1
            if state is not None and unknown_streak <= unknown_hold:
                outputs.append(state)
            else:
                outputs.append("unknown")
            continue
        unknown_streak = 0

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
                outputs.append(state)
        else:
            challenger = None
            challenger_count = 0
            outputs.append("unknown")
    return outputs


def metric_for_config(
    tracks: dict[str, list[int]],
    observations_by_track: dict[str, list[tuple[str, float, float]]],
    truths_by_track: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    final_labels: list[str] = []
    final_truths: list[str] = []
    labels_by_track: dict[str, list[str]] = {}
    for track in tracks:
        sequence = frontier_fuse(observations_by_track[track], **config)
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


def select_frontier(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select maximum safe coverage and retain diagnostic frontier points."""
    if not candidates:
        raise ValueError("candidate list is empty")

    def precision(value):
        return value["track_final"]["precision"]

    def coverage(value):
        return value["track_final"]["coverage"]

    def stability(value):
        return value["stability"]["transition_stability"]

    for value in candidates:
        value["passes"] = precision(value) >= 0.93 and coverage(value) >= 0.45 and stability(value) >= 0.95

    passing = [value for value in candidates if value["passes"]]
    precision_stable = [value for value in candidates if precision(value) >= 0.93 and stability(value) >= 0.95]
    coverage_stable = [value for value in candidates if coverage(value) >= 0.45 and stability(value) >= 0.95]
    stable = [value for value in candidates if stability(value) >= 0.95]

    safe_best = max(
        precision_stable,
        key=lambda value: (coverage(value), precision(value), stability(value)),
    ) if precision_stable else None
    selected = max(
        passing,
        key=lambda value: (coverage(value), precision(value), stability(value)),
    ) if passing else safe_best
    if selected is None:
        selected = max(candidates, key=lambda value: (precision(value), stability(value), coverage(value)))

    pareto = []
    for value in stable:
        dominated = any(
            other is not value
            and precision(other) >= precision(value)
            and coverage(other) >= coverage(value)
            and (precision(other) > precision(value) or coverage(other) > coverage(value))
            for other in stable
        )
        if not dominated:
            pareto.append(value)
    pareto.sort(key=lambda value: (coverage(value), precision(value)))

    def compact(value):
        if value is None:
            return None
        return {
            "track_final": value["track_final"],
            "stability": value["stability"],
            "config": value["config"],
            "passes": value["passes"],
        }

    diagnostics = {
        "candidate_count": len(candidates),
        "passing_count": len(passing),
        "precision_stability_count": len(precision_stable),
        "coverage_stability_count": len(coverage_stable),
        "precision_stability_best": compact(safe_best),
        "coverage_stability_best": compact(max(coverage_stable, key=lambda value: (precision(value), coverage(value))) if coverage_stable else None),
        "maximum_coverage": compact(max(candidates, key=lambda value: (coverage(value), precision(value), stability(value)))),
        "pareto_frontier": [compact(value) for value in pareto[:64]],
    }
    return selected, diagnostics


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
    for window in (1, 3, 5):
        for minimum_observations in (1, 2):
            for entry_share in (0.60, 0.70):
                for entry_margin in (0.0, 0.15):
                    for switch_share in (0.70, 0.80):
                        for switch_margin in (0.15, 0.25):
                            for switch_patience in (2, 3):
                                for unknown_hold in (0, 1, 2):
                                    config = {
                                        "window": window,
                                        "minimum_observations": minimum_observations,
                                        "entry_share": entry_share,
                                        "entry_margin": entry_margin,
                                        "switch_share": switch_share,
                                        "switch_margin": switch_margin,
                                        "switch_patience": switch_patience,
                                        "unknown_hold": unknown_hold,
                                    }
                                    candidates.append(metric_for_config(
                                        tracks, observations_by_track, truths_by_track, config
                                    ))

    selected, diagnostics = select_frontier(candidates)
    selected = dict(selected)
    lengths = Counter(len(indexes) for indexes in tracks.values())
    selected["fusion_search"] = {
        "algorithm": "validation-only causal singleton-aware bounded-hold precision/coverage frontier",
        "selection": "maximum coverage among precision>=0.93 and stability>=0.95; final gate still requires coverage>=0.45",
        **diagnostics,
        "track_length_distribution": {
            "tracks": len(tracks),
            "singleton_tracks": lengths[1],
            "singleton_fraction": lengths[1] / len(tracks) if tracks else 0.0,
            "tracks_lte_2": sum(count for length, count in lengths.items() if length <= 2),
            "counts": {str(length): count for length, count in sorted(lengths.items())},
        },
        "test_accessed": False,
        "frozen_video_used": False,
    }
    return selected


def main() -> int:
    component.track_metrics = track_metrics
    return component.main()


if __name__ == "__main__":
    raise SystemExit(main())
