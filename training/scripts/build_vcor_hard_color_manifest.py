#!/usr/bin/env python3
"""Build a model-independent hard-color subset from the audited VCoR manifest.

The selector deliberately uses only source metadata and image statistics.  It
must not use predictions, confidences, thresholds, or the frozen demo video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--small-area", type=float, default=128 * 96)
    parser.add_argument("--low-gray", type=float, default=64.0)
    parser.add_argument("--high-gray", type=float, default=192.0)
    parser.add_argument("--low-edge", type=float, default=1000.0)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError("input manifest is empty")

    required = {
        "image_path",
        "color",
        "color_supervised",
        "split",
        "review_status",
        "source_dataset",
        "source_license",
        "width",
        "height",
        "gray_mean",
        "edge_variance",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"input manifest is missing required fields: {missing}")

    added = [
        "vehicle_size",
        "lighting",
        "hard_example_priority",
        "hard_mining_tags",
        "hard_score",
        "hard_selector_version",
    ]
    output_fields = fields + [name for name in added if name not in fields]
    selected: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_color_counts: Counter[str] = Counter()
    split_reason_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    for source in rows:
        row = dict(source)
        split = str(row.get("split", "")).strip().lower()
        color = str(row.get("color", "")).strip().lower()
        approved = str(row.get("review_status", "approved")).strip().lower() == "approved"
        supervised = parse_bool(row.get("color_supervised", "true"))
        if split not in {"train", "validation", "test"}:
            rejected["unsupported_split"] += 1
            continue
        if not approved or not supervised or color in {"", "unknown"}:
            rejected["not_approved_color_truth"] += 1
            continue

        width = number(row, "width")
        height = number(row, "height")
        gray = number(row, "gray_mean")
        edge = number(row, "edge_variance")
        if width is None or height is None or width <= 0 or height <= 0:
            rejected["invalid_dimensions"] += 1
            continue

        reasons: list[str] = []
        score = 0
        if parse_bool(row.get("night")):
            reasons.append("night")
            score += 3
        if str(row.get("crop_quality", "")).strip().lower() == "usable":
            reasons.append("usable_quality")
            score += 2
        if width * height < args.small_area:
            reasons.append("small_target")
            score += 2
        if gray is not None and gray < args.low_gray:
            reasons.append("low_light")
            score += 2
        if gray is not None and gray > args.high_gray:
            reasons.append("overexposed")
            score += 1
        if edge is not None and edge < args.low_edge:
            reasons.append("low_detail")
            score += 2
        if parse_bool(row.get("blur")):
            reasons.append("blur")
            score += 2
        if not reasons:
            rejected["not_hard_by_objective_metadata"] += 1
            continue

        area = width * height
        row["vehicle_size"] = "small" if area < args.small_area else "medium" if area < 256 * 192 else "large"
        if parse_bool(row.get("night")):
            row["lighting"] = "night"
        elif gray is None:
            row["lighting"] = "unknown"
        elif gray < args.low_gray:
            row["lighting"] = "low_light"
        elif gray < 128:
            row["lighting"] = "moderate_light"
        else:
            row["lighting"] = "daylight"
        row["hard_example_priority"] = "high" if score >= 4 else "medium"
        row["hard_mining_tags"] = ";".join(reasons)
        row["hard_score"] = str(score)
        row["hard_selector_version"] = "vcor-objective-metadata-v1"
        selected.append(row)
        split_counts[split] += 1
        split_color_counts[f"{split}:{color}"] += 1
        for reason in reasons:
            reason_counts[reason] += 1
            split_reason_counts[f"{split}:{reason}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "schema_version": "vcor-hard-color-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if selected else "fail",
        "policy": {
            "selection_inputs": "source metadata and deterministic image statistics only",
            "model_predictions_used": False,
            "frozen_video_used": False,
            "threshold_or_temperature_selection_used": False,
            "official_split_preserved": True,
            "source_labels_modified": False,
            "criteria_union": {
                "night": True,
                "crop_quality": "usable",
                "small_area_lt": args.small_area,
                "gray_mean_lt": args.low_gray,
                "gray_mean_gt": args.high_gray,
                "edge_variance_lt": args.low_edge,
                "blur": True,
            },
        },
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows_seen": len(rows),
        "rows_selected": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "split_color_counts": dict(sorted(split_color_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "split_reason_counts": dict(sorted(split_reason_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "source_datasets": dict(sorted(Counter(row.get("source_dataset", "") for row in selected).items())),
        "source_licenses": dict(sorted(Counter(row.get("source_license", "") for row in selected).items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
