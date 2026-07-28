#!/usr/bin/env python3
"""Compare native/ONNX/TensorRT metric reports with an absolute-drop gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def compare(reference: dict, candidate: dict, max_drop: float) -> list[str]:
    errors: list[str] = []
    for field in (
        "schema_version",
        "artifact_id",
        "dataset_version",
        "labels_version",
        "sample_count",
    ):
        if reference.get(field) != candidate.get(field):
            errors.append(f"{field} mismatch")
    reference_metrics = reference.get("metrics")
    candidate_metrics = candidate.get("metrics")
    if not isinstance(reference_metrics, dict) or not isinstance(candidate_metrics, dict):
        return errors + ["metrics must be objects"]
    if set(reference_metrics) != set(candidate_metrics):
        errors.append("metric names mismatch")
        return errors
    for name in sorted(reference_metrics):
        base = reference_metrics[name]
        current = candidate_metrics[name]
        if (
            not isinstance(base, (int, float))
            or isinstance(base, bool)
            or not isinstance(current, (int, float))
            or isinstance(current, bool)
        ):
            errors.append(f"metric {name} must be numeric")
            continue
        drop = float(base) - float(current)
        if drop > max_drop + 1e-12:
            errors.append(
                f"metric {name} dropped {drop:.6f}, limit is {max_drop:.6f}"
            )
    if not candidate.get("predictions_sha256"):
        errors.append("candidate predictions_sha256 is required")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-drop", type=float, default=0.005)
    args = parser.parse_args()
    errors = compare(load(args.reference), load(args.candidate), args.max_drop)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        f"PASS: {args.candidate} stays within "
        f"{args.max_drop:.3%} absolute metric drop"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
