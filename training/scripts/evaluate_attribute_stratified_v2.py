#!/usr/bin/env python3
"""Enhanced fixed-threshold attribute evaluation with class/source strata.

The SHA-pinned baseline inference implementation is reused. This wrapper adds
selective per-class precision/recall/coverage and expands metadata strata for
camera, video, weather, blur, reflection, and overlap without changing model
outputs or thresholds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
FROZEN_NAMES = {"vcas_rtsp_demo_60s.mp4", "baseline-preview-36-48s.mp4"}


def ensure_script_import_path() -> None:
    value = str(SCRIPT_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def selective_per_class(
    predictions: list[int],
    targets: list[int],
    confidences: list[float],
    labels: list[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    unknown = labels.index("unknown") if "unknown" in labels else -1
    selected = {
        index
        for index, (prediction, confidence) in enumerate(
            zip(predictions, confidences)
        )
        if prediction != unknown and confidence >= threshold
    }
    result: dict[str, dict[str, Any]] = {}
    for class_id, label in enumerate(labels):
        truth = [index for index, target in enumerate(targets) if target == class_id]
        emitted = [
            index
            for index in selected
            if predictions[index] == class_id
        ]
        correct = [index for index in emitted if targets[index] == class_id]
        truth_selected = [index for index in truth if index in selected]
        support = len(truth)
        result[label] = {
            "truth_support": support,
            "emitted_as_class": len(emitted),
            "correct": len(correct),
            "precision": len(correct) / len(emitted) if emitted else 0.0,
            "recall": len(correct) / support if support else 0.0,
            "coverage": len(truth_selected) / support if support else 0.0,
            "correct_over_all": len(correct) / support if support else 0.0,
            "unknown_class": class_id == unknown,
        }
    return result


def expanded_metadata(base: dict[str, str], row: dict[str, str]) -> dict[str, str]:
    result = dict(base)
    result.update(
        {
            "camera_source": (
                row.get("camera_id") or row.get("camera") or "unknown"
            ),
            "video_source": (
                row.get("video_id") or row.get("sequence_id") or "unknown"
            ),
            "weather": row.get("weather") or "unknown",
            "motion_blur": "blurred" if parse_bool(row.get("blur")) else "clear",
            "reflection": (
                "reflective"
                if parse_bool(row.get("reflection"))
                or parse_bool(row.get("strong_reflection"))
                or parse_bool(row.get("glare"))
                else "not_marked"
            ),
            "vehicle_overlap": (
                "overlap"
                if parse_bool(row.get("overlap"))
                or parse_bool(row.get("vehicle_overlap"))
                else "not_marked"
            ),
        }
    )
    return {key: str(value or "unknown") for key, value in result.items()}


def argument_value(arguments: list[str], name: str) -> str:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"required argument missing: {name}") from error


def validate_fixed_inputs(arguments: list[str]) -> tuple[Path, str]:
    split = argument_value(arguments, "--split")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    output = Path(argument_value(arguments, "--output"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stratified evidence: {output}")
    for name in ("--manifest", "--checkpoint", "--type-threshold", "--color-threshold"):
        value = argument_value(arguments, name)
        if Path(value).name.lower() in FROZEN_NAMES:
            raise ValueError("frozen video artifacts are forbidden inputs")
    return output, split


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    output, split = validate_fixed_inputs(arguments)
    ensure_script_import_path()
    import evaluate_attribute_baseline as baseline

    original_head_metrics = baseline.head_metrics
    original_metadata = baseline.derive_metadata

    def head_metrics_v2(pred, target, confidence, labels, threshold):
        result = original_head_metrics(pred, target, confidence, labels, threshold)
        result["per_class_selective"] = selective_per_class(
            list(pred), list(target), list(confidence), list(labels), float(threshold)
        )
        return result

    def derive_metadata_v2(row, image_path=None):
        return expanded_metadata(original_metadata(row, image_path), row)

    baseline.head_metrics = head_metrics_v2
    baseline.derive_metadata = derive_metadata_v2
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *arguments]
        result = baseline.main()
    finally:
        sys.argv = original_argv
    if result != 0:
        return result

    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("split") != split:
        raise RuntimeError("baseline evaluator returned another split")
    report["schema_version"] = "attribute-stratified-evaluation-v2"
    report["protocol"] = {
        "split": split,
        "fixed_thresholds": True,
        "threshold_search_for_this_report": False,
        "test_used_for_selection": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "per_class_metrics": "selective precision, recall, coverage, and correct-over-all",
        "strata": [
            "crop_quality",
            "occlusion_level",
            "vehicle_size",
            "lighting",
            "viewpoint",
            "source_dataset",
            "label_confidence",
            "camera_source",
            "video_source",
            "weather",
            "motion_blur",
            "reflection",
            "vehicle_overlap"
        ],
        "base_evaluator": str(Path(baseline.__file__).resolve()),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
