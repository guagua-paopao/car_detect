from __future__ import annotations

import random
from collections.abc import Iterable


SPLITS = ("train", "validation", "test")


def assign_group_splits(
    groups: Iterable[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 20260728,
) -> dict[str, str]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")

    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    count = len(unique)
    train_end = int(round(count * train_ratio))
    validation_end = train_end + int(round(count * validation_ratio))
    validation_end = min(validation_end, count)

    result: dict[str, str] = {}
    for index, group in enumerate(unique):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        result[group] = split
    return result


def find_group_leakage(
    rows: Iterable[dict[str, str]],
    group_keys: tuple[str, ...] = ("camera_id", "video_id", "track_group"),
) -> list[str]:
    assignments: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        split = row.get("split", "")
        for key in group_keys:
            value = row.get(key, "")
            if not value:
                errors.append(f"row {row_index}: missing {key}")
                continue
            identity = (key, value)
            previous = assignments.get(identity)
            if previous is not None and previous != split:
                errors.append(
                    f"row {row_index}: {key}={value!r} appears in "
                    f"{previous!r} and {split!r}"
                )
            else:
                assignments[identity] = split
    return errors
