#!/usr/bin/env python3
"""Build model-independent complex-scene attribute splits from a manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hard_tags(row: dict[str, str]) -> list[str]:
    tags: list[str] = []
    lighting = str(row.get("lighting", "")).strip().lower()
    if parse_bool(row.get("night")) or lighting in {"night", "low_light"}:
        tags.append("night_or_low_light")
    if str(row.get("vehicle_size", "")).strip().lower() == "small" or parse_bool(row.get("small_target")):
        tags.append("small_target")
    if parse_bool(row.get("blur")):
        tags.append("blur")
    occlusion = str(row.get("occlusion_level", "")).strip().lower()
    if parse_bool(row.get("occluded")) or parse_bool(row.get("truncated")) or occlusion in {"occluded", "truncated", "heavy"}:
        tags.append("occluded_or_truncated")
    return tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    args = parser.parse_args()

    allowed_splits = {value.strip().lower() for value in args.splits}
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not fields:
        raise RuntimeError("input manifest is empty")

    output_fields = fields + ([] if "hard_selection_tags" in fields else ["hard_selection_tags"])
    selected: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    split_body_supervised: Counter[str] = Counter()
    split_color_supervised: Counter[str] = Counter()
    split_tag_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for source in rows:
        split = str(source.get("split", "")).strip().lower()
        if split not in allowed_splits:
            continue
        tags = hard_tags(source)
        if not tags:
            continue
        row = dict(source)
        row["hard_selection_tags"] = ";".join(tags)
        selected.append(row)
        split_counts[split] += 1
        source_counts[f"{split}:{row.get('source_dataset', 'unknown')}"] += 1
        if (
            parse_bool(row.get("body_type_supervised", "true"))
            and str(row.get("body_type", "")).strip().lower() not in {"", "unknown"}
        ):
            split_body_supervised[split] += 1
        if (
            parse_bool(row.get("color_supervised", "true"))
            and str(row.get("color", "")).strip().lower() not in {"", "unknown"}
        ):
            split_color_supervised[split] += 1
        for tag in tags:
            split_tag_counts[f"{split}:{tag}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "schema_version": "attribute-hard-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if selected else "fail",
        "policy": {
            "selection_inputs": "manifest scene metadata only",
            "model_predictions_used": False,
            "frozen_video_used": False,
            "threshold_or_temperature_selection_used": False,
            "source_splits_preserved": True,
            "selected_splits": sorted(allowed_splits),
            "criteria_union": [
                "night_or_low_light",
                "small_target",
                "blur",
                "occluded_or_truncated",
            ],
        },
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "rows_seen": len(rows),
        "rows_selected": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "split_body_supervised": dict(sorted(split_body_supervised.items())),
        "split_color_supervised": dict(sorted(split_color_supervised.items())),
        "split_tag_counts": dict(sorted(split_tag_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
