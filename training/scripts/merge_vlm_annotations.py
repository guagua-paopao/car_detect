#!/usr/bin/env python3
"""Merge calibrated VLM prelabels without changing the closed validation/test set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, write_json  # noqa: E402


FIELDS = (
    "body_type",
    "color",
    "crop_quality",
    "viewpoint",
    "blur",
    "occluded",
    "truncated",
    "night",
)
CALIBRATED_FIELDS = ("body_type", "color")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        nargs="+",
        help="one or more resumable teacher JSONL files",
    )
    parser.add_argument(
        "--secondary-predictions-csv",
        type=Path,
        help="optional independent classifier predictions used for consensus",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--materialize-crops",
        action="store_true",
        help="hard-link crops beside the output manifest, falling back to copies",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--target-precision", type=float, default=0.93)
    parser.add_argument("--minimum-calibration-samples", type=int, default=10)
    parser.add_argument("--secondary-body-min-confidence", type=float, default=0.0)
    parser.add_argument("--secondary-color-min-confidence", type=float, default=0.0)
    parser.add_argument("--secondary-quality-min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--teacher-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    return parser.parse_args()


def read_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                image_path = str(value.get("image_path", ""))
                if not image_path:
                    raise ValueError(f"{path}:{line_number}: image_path is missing")
                values[image_path] = value
    return values


def read_secondary_predictions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["image_path"]: row
            for row in csv.DictReader(handle)
            if row.get("image_path")
        }


def secondary_agrees(
    secondary: dict[str, str] | None,
    prediction: dict[str, Any],
    field: str,
    minimum_confidence: dict[str, float],
    required: bool,
) -> bool:
    if secondary is None:
        return not required
    try:
        confidence = float(secondary[f"{field}_confidence"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        secondary.get(field) == prediction.get(field)
        and confidence >= minimum_confidence[field]
    )


def calibrate_threshold(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    secondary_predictions: dict[str, dict[str, str]],
    secondary_minimum_confidence: dict[str, float],
    require_secondary: bool,
    field: str,
    target_precision: float,
    minimum_samples: int,
) -> dict[str, Any]:
    examples: list[tuple[float, bool]] = []
    for row in rows:
        if (
            row.get("review_status") != "approved"
            or row.get("split") != "train"
        ):
            continue
        record = predictions.get(row["image_path"])
        prediction = record.get("prediction") if record else None
        if not isinstance(prediction, dict):
            continue
        if prediction.get(field) == "unknown":
            continue
        if not secondary_agrees(
            secondary_predictions.get(row["image_path"]),
            prediction,
            field,
            secondary_minimum_confidence,
            require_secondary,
        ):
            continue
        if require_secondary:
            confidence = float(
                secondary_predictions[row["image_path"]][f"{field}_confidence"]
            )
        else:
            confidence = float(prediction.get("confidence", {}).get(field, 0.0))
        examples.append((confidence, prediction.get(field) == row.get(field)))
    candidates = sorted(
        {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99}
        | {confidence for confidence, _ in examples}
    )
    selected_threshold: float | None = None
    selected_count = 0
    selected_precision: float | None = None
    table: list[dict[str, Any]] = []
    for threshold in candidates:
        chosen = [correct for confidence, correct in examples if confidence >= threshold]
        precision = sum(chosen) / len(chosen) if chosen else None
        table.append(
            {
                "threshold": threshold,
                "selected": len(chosen),
                "precision": precision,
            }
        )
        if (
            selected_threshold is None
            and len(chosen) >= minimum_samples
            and precision is not None
            and precision >= target_precision
        ):
            selected_threshold = threshold
            selected_count = len(chosen)
            selected_precision = precision
    return {
        "threshold": selected_threshold,
        "selected": selected_count,
        "precision": selected_precision,
        "calibration_examples": len(examples),
        "table": table,
    }


def normalized_prediction(
    prediction: dict[str, Any],
    labels: dict[str, Any],
) -> bool:
    return (
        prediction.get("body_type") in set(labels["body_types"])
        and prediction.get("color") in set(labels["colors"])
        and prediction.get("crop_quality") in set(labels["crop_qualities"])
        and prediction.get("viewpoint") in set(labels["viewpoints"])
        and all(str(prediction.get(field)).lower() in {"true", "false"} for field in FIELDS[4:])
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_crops(
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    if source_root == output_root:
        return
    for row in rows:
        relative = Path(str(row["image_path"]))
        if relative.is_absolute():
            raise ValueError("image_path must stay relative when materializing crops")
        source = source_root / relative
        target = output_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing source crop: {source}")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        manifest_fields = list(reader.fieldnames)
        rows = list(reader)
    predictions = read_predictions(args.predictions)
    secondary_predictions = read_secondary_predictions(
        args.secondary_predictions_csv
    )
    secondary_minimum_confidence = {
        "body_type": args.secondary_body_min_confidence,
        "color": args.secondary_color_min_confidence,
        "crop_quality": args.secondary_quality_min_confidence,
    }
    require_secondary = args.secondary_predictions_csv is not None
    calibration = {
        field: calibrate_threshold(
            rows,
            predictions,
            secondary_predictions,
            secondary_minimum_confidence,
            require_secondary,
            field,
            args.target_precision,
            args.minimum_calibration_samples,
        )
        for field in CALIBRATED_FIELDS
    }
    missing_thresholds = [
        field for field, value in calibration.items() if value["threshold"] is None
    ]
    if missing_thresholds:
        raise RuntimeError(
            "teacher did not reach the requested calibrated precision for "
            f"{missing_thresholds}; no pseudo labels were accepted"
        )

    provenance_fields = [
        "annotation_source",
        "annotation_teacher",
        "body_type_supervised",
        "color_supervised",
        "body_type_confidence",
        "color_confidence",
        "crop_quality_confidence",
    ]
    output_fields = manifest_fields + [
        field for field in provenance_fields if field not in manifest_fields
    ]
    review_fields = output_fields + [
        "suggested_body_type",
        "suggested_color",
        "suggested_crop_quality",
        "review_reason",
    ]
    output_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    accepted = 0
    preserved = 0
    reasons: Counter[str] = Counter()

    for source in rows:
        row = dict(source)
        if row.get("review_status") == "approved":
            row["annotation_source"] = row.get("annotation_source") or "human"
            row["body_type_supervised"] = "true"
            row["color_supervised"] = "true"
            output_rows.append(row)
            preserved += 1
            continue
        row["body_type_supervised"] = "false"
        row["color_supervised"] = "false"
        record = predictions.get(row["image_path"])
        prediction = record.get("prediction") if record else None
        reason = ""
        if row.get("split") != "train":
            reason = "closed_validation_or_test_requires_manual_review"
        elif not isinstance(prediction, dict):
            reason = "missing_or_invalid_teacher_prediction"
        elif not normalized_prediction(prediction, labels):
            reason = "prediction_violates_local_label_contract"
        else:
            confidence = prediction["confidence"]
            passed: dict[str, bool] = {}
            for field in CALIBRATED_FIELDS:
                secondary = secondary_predictions.get(row["image_path"])
                score = (
                    float(secondary[f"{field}_confidence"])
                    if require_secondary and secondary is not None
                    else float(confidence.get(field, 0.0))
                )
                passed[field] = (
                    prediction.get(field) != "unknown"
                    and
                    score >= float(calibration[field]["threshold"])
                    and secondary_agrees(
                        secondary,
                        prediction,
                        field,
                        secondary_minimum_confidence,
                        require_secondary,
                    )
                )
            if not any(passed.values()):
                reason = "below_calibrated_precision_threshold"
            else:
                for field in FIELDS:
                    row[field] = str(prediction[field]).lower()
                for field in CALIBRATED_FIELDS:
                    if not passed[field]:
                        row[field] = "unknown"
                secondary = secondary_predictions.get(row["image_path"])
                quality_agrees = secondary_agrees(
                    secondary,
                    prediction,
                    "crop_quality",
                    secondary_minimum_confidence,
                    require_secondary,
                )
                if (
                    all(passed.values())
                    and quality_agrees
                    and prediction["crop_quality"] == "good"
                ):
                    row["crop_quality"] = "good"
                else:
                    row["crop_quality"] = "usable"
                row["review_status"] = "approved"
                row["annotation_source"] = "calibrated_vlm"
                row["annotation_teacher"] = args.teacher_id
                for field in CALIBRATED_FIELDS:
                    row[f"{field}_supervised"] = str(passed[field]).lower()
                    secondary = secondary_predictions.get(row["image_path"])
                    row[f"{field}_confidence"] = str(
                        float(secondary[f"{field}_confidence"])
                        if require_secondary and secondary is not None
                        else float(confidence.get(field, 0.0))
                    )
                row["crop_quality_confidence"] = str(
                    float(
                        secondary["crop_quality_confidence"]
                        if require_secondary and secondary is not None
                        else confidence.get("crop_quality", 0.0)
                    )
                )
                output_rows.append(row)
                accepted += 1
                continue

        reasons[reason] += 1
        output_rows.append(row)
        queued = dict(row)
        if isinstance(prediction, dict):
            queued["suggested_body_type"] = prediction.get("body_type", "")
            queued["suggested_color"] = prediction.get("color", "")
            queued["suggested_crop_quality"] = prediction.get("crop_quality", "")
        queued["review_reason"] = reason
        review_rows.append(queued)

    if args.materialize_crops:
        materialize_crops(
            args.manifest.resolve().parent,
            args.output.resolve().parent,
            output_rows,
        )
    write_csv(args.output.resolve(), output_fields, output_rows)
    write_csv(args.review_queue.resolve(), review_fields, review_rows)
    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "teacher_id": args.teacher_id,
        "secondary_predictions_csv": (
            str(args.secondary_predictions_csv.resolve())
            if args.secondary_predictions_csv
            else None
        ),
        "secondary_minimum_confidence": secondary_minimum_confidence,
        "target_precision": args.target_precision,
        "calibration_policy": "approved_train_only",
        "calibration": calibration,
        "rows": len(rows),
        "human_approved_preserved": preserved,
        "auto_approved_train": accepted,
        "manual_review_queue": len(review_rows),
        "review_reasons": dict(reasons),
        "output": str(args.output.resolve()),
        "review_queue": str(args.review_queue.resolve()),
    }
    write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
