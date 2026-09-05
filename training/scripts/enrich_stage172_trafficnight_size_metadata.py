#!/usr/bin/env python3
"""Add conservative source-frame small-target metadata to sealed Stage172 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXTRA_FIELDS = [
    "source_bbox_area_ratio", "source_box_width_pixels", "source_box_height_pixels",
    "vehicle_size", "small_target", "stage172_size_truth_source",
    "small_target_proxy_area_threshold",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bbox_metrics(row: dict[str, str]) -> tuple[float, float, float]:
    points = json.loads(row["stage172_polygon_points"])
    dimensions = row["stage172_decoded_dimensions"].lower().split("x")
    if len(dimensions) != 2:
        raise ValueError("invalid decoded dimensions")
    width, height = int(dimensions[0]), int(dimensions[1])
    if width <= 0 or height <= 0 or not isinstance(points, list) or len(points) < 3:
        raise ValueError("invalid polygon or decoded dimensions")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    if box_width <= 0 or box_height <= 0:
        raise ValueError("degenerate source-frame box")
    ratio = box_width * box_height / (width * height)
    if not 0 < ratio <= 1:
        raise ValueError("invalid source-frame box ratio")
    return box_width, box_height, ratio


def enrich(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    source = args.source.resolve()
    source_audit = args.source_audit.resolve()
    actual_source_sha = sha256_file(source)
    actual_audit_sha = sha256_file(source_audit)
    if actual_source_sha != args.expected_source_sha256.lower():
        raise RuntimeError("source manifest SHA256 mismatch")
    if actual_audit_sha != args.expected_source_audit_sha256.lower():
        raise RuntimeError("source audit SHA256 mismatch")
    audit = json.loads(source_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("failures"):
        raise RuntimeError("source crop audit did not pass")
    if str(audit.get("inputs", {}).get("manifest_sha256", "")).lower() != actual_source_sha:
        raise RuntimeError("source audit manifest lineage mismatch")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError("source manifest is empty")
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)

    counts: Counter[str] = Counter()
    ratios: list[float] = []
    for row in rows:
        if row.get("split") != "train":
            raise RuntimeError("Stage172 size enrichment accepts train rows only")
        if row.get("color") != "unknown" or row.get("color_supervised") != "false":
            raise RuntimeError("Stage172 size enrichment found color supervision")
        box_width, box_height, ratio = bbox_metrics(row)
        small = ratio <= args.small_area_ratio
        row["source_bbox_area_ratio"] = f"{ratio:.9f}"
        row["source_box_width_pixels"] = f"{box_width:.3f}"
        row["source_box_height_pixels"] = f"{box_height:.3f}"
        row["vehicle_size"] = "small" if small else "medium_or_large"
        row["small_target"] = str(small).lower()
        row["stage172_size_truth_source"] = "official_polygon_bbox_ratio_in_decoded_source_frame"
        row["small_target_proxy_area_threshold"] = f"{args.small_area_ratio:.6f}"
        ratios.append(ratio)
        counts["small" if small else "medium_or_large"] += 1
        counts[f"source_label:{row.get('source_label', '')}"] += 1
        if small and (row.get("body_type_supervised") == "true" or row.get("coarse_body_family") in {"car", "truck"}):
            counts["small_with_fine_or_coarse_body_truth"] += 1

    report = {
        "schema_version": "stage172-trafficnight-size-metadata-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "source": str(source), "source_sha256": actual_source_sha,
            "source_audit": str(source_audit), "source_audit_sha256": actual_audit_sha,
        },
        "policy": {
            "metric": "axis-aligned official polygon bounding-box area divided by decoded source-frame area",
            "small_area_ratio_threshold": args.small_area_ratio,
            "source_viewpoint": "aerial_oblique_night_auxiliary",
            "small_target_is_source_frame_proxy": True,
            "small_target_is_not_gate_camera_validation_truth": True,
            "image_bytes_modified": False,
            "labels_modified": False,
            "all_rows_train_only": True,
            "all_colors_unknown": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
        "output": {
            "rows": len(rows),
            "counts": dict(sorted(counts.items())),
            "small_fraction": counts["small"] / len(rows),
            "area_ratio_min": min(ratios),
            "area_ratio_median": sorted(ratios)[len(ratios) // 2],
            "area_ratio_max": max(ratios),
        },
        "failures": [],
    }
    return fields, rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    parser.add_argument("--small-area-ratio", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage172 size evidence")
    if not 0 < args.small_area_ratio < 1:
        raise ValueError("small area ratio must be in (0, 1)")
    fields, rows, report = enrich(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["output"].update({
        "manifest": str(args.output.resolve()),
        "manifest_sha256": sha256_file(args.output),
    })
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(rows), "small": report["output"]["counts"]["small"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
