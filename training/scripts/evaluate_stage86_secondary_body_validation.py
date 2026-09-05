#!/usr/bin/env python3
"""Apply Stage85's frozen validation thresholds to secondary hard/weather views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_primary_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage85_primary_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load primary evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_thresholds(report: dict[str, Any]) -> dict[str, float]:
    if report.get("status") != "complete_validation_only":
        raise RuntimeError("Stage85 primary report is not complete")
    selection = report.get("threshold_selection", {})
    if selection.get("source") != "VFG-7 validation only":
        raise RuntimeError("Stage85 threshold source mismatch")
    gates = report.get("shared_validation_gates", {})
    if gates.get("all_pass") is not True:
        raise RuntimeError("Stage85 primary gates did not pass")
    values = {
        "candidate_body": float(selection.get("body_exact", {}).get("threshold", 0.0)),
        "candidate_body_family": float(selection.get("body_family", {}).get("threshold", 0.0)),
        "baseline_body": float(selection.get("baseline_body", {}).get("threshold", 0.0)),
    }
    if any(not 0.0 < value <= 1.0 for value in values.values()):
        raise RuntimeError("Stage85 threshold evidence is invalid")
    return values


def body_vectors(
    module: ModuleType,
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
) -> tuple[list[int], list[str], list[float], list[str]]:
    indexes = [index for index, row in enumerate(rows) if module.truthy(row.get("body_type_supervised"))]
    return (
        indexes,
        [outputs[index]["body_label"] for index in indexes],
        [float(outputs[index]["body_confidence"]) for index in indexes],
        [rows[index]["body_type"] for index in indexes],
    )


def stratified_body(
    module: ModuleType,
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        "lighting", "vehicle_size", "weather", "occlusion_level",
        "source_dataset", "camera_id", "video_id",
    )
    for field in fields:
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if module.truthy(row.get("body_type_supervised")):
                groups[str(row.get(field) or "unknown")].append(index)
        result[field] = {}
        for key, indexes in sorted(groups.items()):
            result[field][key] = module.metric_at_threshold(
                [outputs[index]["body_label"] for index in indexes],
                [outputs[index]["body_confidence"] for index in indexes],
                [rows[index]["body_type"] for index in indexes],
                threshold,
            )
    return result


def evaluate_view(
    module: ModuleType,
    args: argparse.Namespace,
    manifest: Path,
    body_checkpoint: Path,
    color_checkpoint: Path,
    baseline_checkpoint: Path,
    labels: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    rows = module.resolve_rows(manifest, args.datasets_safety_root)
    candidate = module.infer_pair(
        args, rows, body_checkpoint, color_checkpoint, expected_labels=labels
    )
    baseline = module.infer_pair(
        args, rows, baseline_checkpoint, baseline_checkpoint, expected_labels=None
    )
    indexes, predictions, confidences, truths = body_vectors(module, rows, candidate)
    _, baseline_predictions, baseline_confidences, _ = body_vectors(module, rows, baseline)
    if not indexes:
        raise RuntimeError(f"secondary view has no body truth: {manifest}")
    complex_positions = [position for position, index in enumerate(indexes) if module.is_complex_row(rows[index])]
    if not complex_positions:
        raise RuntimeError(f"secondary view has no complex body truth: {manifest}")
    exact = module.metric_at_threshold(
        predictions, confidences, truths, thresholds["candidate_body"]
    )
    family = module.metric_at_threshold(
        predictions, confidences, truths, thresholds["candidate_body_family"], family_aware=True
    )
    baseline_exact = module.metric_at_threshold(
        baseline_predictions, baseline_confidences, truths, thresholds["baseline_body"]
    )
    complex_exact = module.metric_at_threshold(
        [predictions[position] for position in complex_positions],
        [confidences[position] for position in complex_positions],
        [truths[position] for position in complex_positions],
        thresholds["candidate_body"],
    )
    baseline_complex = module.metric_at_threshold(
        [baseline_predictions[position] for position in complex_positions],
        [baseline_confidences[position] for position in complex_positions],
        [truths[position] for position in complex_positions],
        thresholds["baseline_body"],
    )
    return {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "rows": len(rows),
        "body_supervised": len(indexes),
        "complex_body_supervised": len(complex_positions),
        "candidate": {
            "body_exact": exact,
            "body_family": family,
            "complex_body_exact": complex_exact,
            "per_class": module.per_class_metrics(
                predictions, confidences, truths, thresholds["candidate_body"]
            ),
            "stratified": stratified_body(module, rows, candidate, thresholds["candidate_body"]),
        },
        "production_baseline": {
            "body_exact": baseline_exact,
            "complex_body_exact": baseline_complex,
            "per_class": module.per_class_metrics(
                baseline_predictions, baseline_confidences, truths, thresholds["baseline_body"]
            ),
        },
        "comparison": {
            "complex_coverage_gain": complex_exact["coverage"] - baseline_complex["coverage"],
        },
    }


def secondary_gates(hard: dict[str, Any], weather: dict[str, Any]) -> dict[str, bool]:
    hard_exact = hard["candidate"]["body_exact"]
    hard_complex = hard["candidate"]["complex_body_exact"]
    weather_exact = weather["candidate"]["body_exact"]
    return {
        "hard_body_precision": hard_exact["precision"] >= 0.93,
        "hard_body_coverage": hard_exact["coverage"] >= 0.45,
        "hard_complex_precision": hard_complex["precision"] >= 0.93,
        "hard_complex_coverage_gain": hard["comparison"]["complex_coverage_gain"] >= 0.15,
        "weather_body_precision": weather_exact["precision"] >= 0.93,
        "weather_body_coverage": weather_exact["coverage"] >= 0.25,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-evaluator", type=Path, required=True)
    parser.add_argument("--expected-primary-evaluator-sha256", required=True)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--expected-primary-report-sha256", required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--expected-hard-manifest-sha256", required=True)
    parser.add_argument("--weather-manifest", type=Path, required=True)
    parser.add_argument("--expected-weather-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-color-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite secondary validation evidence: {args.output}")
    pinned = (
        (args.primary_evaluator, args.expected_primary_evaluator_sha256),
        (args.primary_report, args.expected_primary_report_sha256),
        (args.hard_manifest, args.expected_hard_manifest_sha256),
        (args.weather_manifest, args.expected_weather_manifest_sha256),
        (args.labels, args.expected_labels_sha256),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256),
        (args.color_checkpoint, args.expected_color_checkpoint_sha256),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256),
    )
    for path, expected in pinned:
        if not path.is_file() or sha256(path).lower() != expected.lower():
            raise RuntimeError(f"immutable secondary validation input mismatch: {path}")
    module = load_primary_module(args.primary_evaluator)
    primary_report = read_json(args.primary_report)
    thresholds = extract_thresholds(primary_report)
    labels = read_json(args.labels)
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 labels are required")
    hard = evaluate_view(
        module, args, args.hard_manifest, args.body_checkpoint, args.color_checkpoint,
        args.baseline_checkpoint, labels, thresholds,
    )
    weather = evaluate_view(
        module, args, args.weather_manifest, args.body_checkpoint, args.color_checkpoint,
        args.baseline_checkpoint, labels, thresholds,
    )
    gates = secondary_gates(hard, weather)
    report = {
        "schema_version": "stage86-secondary-body-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "threshold_policy": {
            "source": "immutable Stage85 VFG-7 validation report",
            "selection_or_tuning_on_secondary_views": False,
            "values": thresholds,
        },
        "inputs": {
            "primary_report": str(args.primary_report.resolve()),
            "primary_report_sha256": sha256(args.primary_report),
            "body_checkpoint_sha256": sha256(args.body_checkpoint),
            "color_checkpoint_sha256": sha256(args.color_checkpoint),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        },
        "views": {"hard": hard, "weather": weather},
        "secondary_gates": {
            "gates": gates,
            "all_pass": all(gates.values()),
            "decision": (
                "secondary_validation_pass_pending_test_backend_and_external_exact_truth"
                if all(gates.values())
                else "candidate_rejected_secondary_validation_gate_failure"
            ),
        },
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": report["status"],
        "hard": hard["candidate"]["body_exact"],
        "weather": weather["candidate"]["body_exact"],
        "secondary_gates_pass": report["secondary_gates"]["all_pass"],
        "test_accessed": False,
        "frozen_video_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
