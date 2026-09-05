#!/usr/bin/env python3
"""Fail-closed quality-weighted temporal fusion primitives.

This module is intentionally inference-only and contains no data/model access.
It can be compared with the existing stateless sliding-window fusion without
changing a running training or validation pipeline.
"""

from __future__ import annotations

from collections import defaultdict


def _window_candidate(
    history: list[dict],
    minimum_share: float,
    minimum_margin: float,
) -> tuple[str, dict[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for value in history:
        label = str(value.get("label", "unknown") or "unknown")
        if label == "unknown":
            continue
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0) or 0.0)))
        quality = max(0.0, min(1.0, float(value.get("quality", 0.0) or 0.0)))
        scores[label] += confidence * quality
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    total = sum(scores.values())
    if not ranked or total <= 0:
        return "unknown", dict(scores)
    share = ranked[0][1] / total
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (ranked[0][1] - runner_up) / total
    if share < minimum_share or margin < minimum_margin:
        return "unknown", dict(scores)
    return ranked[0][0], dict(scores)


def fuse_sequence_fail_closed(
    values: list[dict],
    *,
    window: int,
    minimum_share: float,
    minimum_margin: float,
    switch_confirmations: int = 3,
    conflict_unknown_after: int = 2,
    override_ratio: float = 1.15,
) -> list[str]:
    """Fuse a single ordered track while refusing unsafe label switches.

    A new label must dominate the retained label's quality-weighted evidence and
    repeat for ``switch_confirmations`` consecutive windows. The first strong
    conflict keeps the trusted history; continued conflict emits ``unknown``
    until the switch is confirmed. Ambiguous windows also become ``unknown``
    after ``conflict_unknown_after`` consecutive occurrences.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if switch_confirmations < 1:
        raise ValueError("switch_confirmations must be positive")
    if conflict_unknown_after < 1:
        raise ValueError("conflict_unknown_after must be positive")
    if not 0.0 <= minimum_share <= 1.0 or not 0.0 <= minimum_margin <= 1.0:
        raise ValueError("share and margin must be in [0, 1]")
    if override_ratio < 1.0:
        raise ValueError("override_ratio must be at least 1")

    output: list[str] = []
    stable_label: str | None = None
    pending_label: str | None = None
    pending_count = 0
    conflict_count = 0

    for index in range(len(values)):
        history = values[max(0, index - window + 1):index + 1]
        candidate, scores = _window_candidate(history, minimum_share, minimum_margin)

        if stable_label is None:
            if candidate == "unknown":
                output.append("unknown")
                continue
            stable_label = candidate
            pending_label = None
            pending_count = 0
            conflict_count = 0
            output.append(stable_label)
            continue

        if candidate == stable_label:
            pending_label = None
            pending_count = 0
            conflict_count = 0
            output.append(stable_label)
            continue

        if candidate == "unknown":
            pending_label = None
            pending_count = 0
            conflict_count += 1
            output.append(stable_label if conflict_count < conflict_unknown_after else "unknown")
            continue

        candidate_score = scores.get(candidate, 0.0)
        stable_score = scores.get(stable_label, 0.0)
        if stable_score > 0.0 and candidate_score < stable_score * override_ratio:
            pending_label = None
            pending_count = 0
            conflict_count = 0
            output.append(stable_label)
            continue

        if pending_label == candidate:
            pending_count += 1
        else:
            pending_label = candidate
            pending_count = 1
        conflict_count += 1
        if pending_count >= switch_confirmations:
            stable_label = candidate
            pending_label = None
            pending_count = 0
            conflict_count = 0
            output.append(stable_label)
        elif conflict_count >= conflict_unknown_after:
            output.append("unknown")
        else:
            output.append(stable_label)
    return output
