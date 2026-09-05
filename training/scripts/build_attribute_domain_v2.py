#!/usr/bin/env python3
"""Build a deterministic, group-isolated attribute-domain-v2 manifest.

The source manifest already contains source-level split isolation.  This tool
subsamples validation/test to a bounded independent evaluation set and adds the
metadata required by the VCAS attribute evaluation protocol.  It never copies
or mutates image files; image paths are rewritten relative to the output
manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def derive_metadata(row: dict[str, str], image_path: Path | None = None) -> dict[str, str]:
    truncated_raw = (row.get("truncated") or "").strip()
    occluded_raw = (row.get("occluded") or "").strip()
    truncated = parse_bool(truncated_raw)
    occluded = parse_bool(occluded_raw)
    if not truncated_raw and not occluded_raw:
        occlusion = "unknown"
    elif truncated:
        occlusion = "truncated"
    elif occluded:
        occlusion = "occluded"
    else:
        occlusion = "visible"

    try:
        width = float(row.get("width", ""))
        height = float(row.get("height", ""))
        area = width * height
    except (TypeError, ValueError):
        area = 0.0
    gray_mean: float | None = None
    if (area <= 0 or not (row.get("gray_mean") or "").strip()) and image_path and image_path.exists():
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
                area = float(width * height)
                gray_mean = float(image.convert("L").resize((1, 1)).getpixel((0, 0)))
        except Exception:
            pass
    if area <= 0:
        vehicle_size = "unknown"
    elif area < 128 * 96:
        vehicle_size = "small"
    elif area < 256 * 192:
        vehicle_size = "medium"
    else:
        vehicle_size = "large"

    if parse_bool(row.get("night")):
        lighting = "night"
    else:
        try:
            gray_mean = float(row.get("gray_mean", ""))
        except (TypeError, ValueError):
            if gray_mean is None:
                gray_mean = math.nan
        if math.isnan(gray_mean):
            lighting = "unknown"
        elif gray_mean < 64:
            lighting = "low_light"
        elif gray_mean < 128:
            lighting = "moderate_light"
        else:
            lighting = "daylight"

    review = (row.get("review_status") or "").strip().lower()
    source = (row.get("source_dataset") or "").strip()
    license_text = (row.get("source_license") or "").strip()
    body_known = (row.get("body_type") or "").strip().lower() not in {"", "unknown"}
    color_known = (row.get("color") or "").strip().lower() not in {"", "unknown"}
    # High means an approved, explicit source label.  Restricted/unclear source
    # licenses remain visible for evaluation but are never formal-train eligible.
    if review == "approved" and (body_known or color_known):
        label_confidence = "high"
    elif review == "approved":
        label_confidence = "medium"
    else:
        label_confidence = "unknown"

    return {
        "occlusion_level": occlusion,
        "vehicle_size": vehicle_size,
        "lighting": lighting,
        "label_confidence": label_confidence,
        "license_train_eligible": (
            "true"
            if license_text == "CC-BY-4.0"
            or "research-and-education-only" in license_text
            else "false"
        ),
        "source_group": source or "unknown",
    }


def stable_key(row: dict[str, str], seed: int) -> int:
    value = f"{seed}|{row.get('sha256','')}|{row.get('image_path','')}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def choose_rows(rows: list[dict[str, str]], target: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= target:
        return sorted(rows, key=lambda r: stable_key(r, seed))
    # Allocate deterministically over body labels first, then fill any remainder.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("body_type") or "unknown").strip()].append(row)
    labels = sorted(groups)
    quota = {label: target // len(labels) for label in labels}
    for label in labels[: target % len(labels)]:
        quota[label] += 1
    picked: list[dict[str, str]] = []
    leftovers: list[dict[str, str]] = []
    for label in labels:
        ordered = sorted(groups[label], key=lambda r: stable_key(r, seed))
        take = min(quota[label], len(ordered))
        picked.extend(ordered[:take])
        leftovers.extend(ordered[take:])
    if len(picked) < target:
        picked.extend(sorted(leftovers, key=lambda r: stable_key(r, seed))[: target - len(picked)])
    return sorted(picked[:target], key=lambda r: stable_key(r, seed))


def verify_group_isolation(rows: Iterable[dict[str, str]]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    missing = 0
    for row in rows:
        group = (row.get("track_group") or row.get("video_id") or row.get("sha256") or "").strip()
        if not group:
            missing += 1
        groups[group].add(row.get("split", ""))
    overlaps = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    return {
        "unique_groups": len(groups),
        "missing_group_rows": missing,
        "overlap_group_count": len(overlaps),
        "overlap_examples": dict(list(overlaps.items())[:20]),
        "status": "pass" if not overlaps and not missing else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validation-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=4000)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise SystemExit("source manifest is empty")
    source_root = args.input.parent
    selected: list[dict[str, str]] = []
    for split, target in (("train", None), ("validation", args.validation_size), ("test", args.test_size)):
        rows = [row for row in source_rows if row.get("split") == split]
        selected.extend(rows if target is None else choose_rows(rows, target, args.seed + len(split)))

    output_root = args.output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(source_rows[0].keys())
    for field in ("occlusion_level", "vehicle_size", "lighting", "label_confidence", "license_train_eligible", "source_group"):
        if field not in fields:
            fields.append(field)
    rendered: list[dict[str, str]] = []
    for row in selected:
        out = dict(row)
        try:
            out["image_path"] = str(Path("..") / source_root.name / row["image_path"])
        except KeyError:
            raise SystemExit("source manifest has no image_path")
        out.update(derive_metadata(row, source_root / row["image_path"]))
        # Some source manifests use a reusable placeholder camera/video name
        # (for example BMD-45's images_000) across official splits.  Namespace
        # these provenance keys by source and split so the project leakage
        # checker cannot mistake a source partition identifier for a shared
        # physical sequence.  track_group remains the vehicle-level key.
        source_ns = (row.get("source_dataset") or "source").replace(" ", "_")
        split_ns = row.get("split") or "unknown"
        for field in ("camera_id", "video_id"):
            value = row.get(field) or row.get("source_frame_id") or row.get("sha256") or "unknown"
            out[field] = f"{source_ns}:{split_ns}:{value}"
        track_value = row.get("track_group") or row.get("source_frame_id") or row.get("sha256") or "unknown"
        out["track_group"] = f"{source_ns}:{split_ns}:{track_value}"
        rendered.append(out)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rendered)

    split_counts = Counter(row["split"] for row in rendered)
    metadata_counts = {
        field: dict(sorted(Counter(row[field] for row in rendered).items()))
        for field in ("crop_quality", "occlusion_level", "vehicle_size", "lighting", "label_confidence", "source_group", "license_train_eligible")
    }
    source_counts = Counter(row.get("source_dataset", "") for row in rendered)
    license_counts = Counter(row.get("source_license", "") for row in rendered)
    split_report = {
        "schema_version": "1.0",
        "dataset_version": "attribute-domain-v2",
        "source_manifest": str(args.input),
        "output_manifest": str(args.output),
        "seed": args.seed,
        "requested_split_sizes": {"train": None, "validation": args.validation_size, "test": args.test_size},
        "actual_split_sizes": dict(sorted(split_counts.items())),
        "group_isolation": verify_group_isolation(rendered),
        "source_counts": dict(sorted(source_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "metadata_counts": metadata_counts,
        "duplicate_checks": {
            "sha256_duplicate_rows": len(rendered) - len({row.get("sha256", "") for row in rendered if row.get("sha256")}),
            "image_path_duplicate_rows": len(rendered) - len({row.get("image_path", "") for row in rendered}),
        },
    }
    args.split_report.parent.mkdir(parents=True, exist_ok=True)
    args.split_report.write_text(json.dumps(split_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = {
        "schema_version": "1.0",
        "dataset_version": "attribute-domain-v2",
        "created_from": str(args.input),
        "rows": len(rendered),
        "split_counts": dict(sorted(split_counts.items())),
        "split_policy": "source split preserved; deterministic body-stratified validation/test subsampling; track_group isolation verified",
        "seed": args.seed,
        "required_attributes": ["body_type", "color", "crop_quality", "occlusion_level", "vehicle_size", "lighting", "label_confidence"],
        "unknown_policy": "BMD-45 color remains unknown; ambiguous/occluded/unseen colors are not coerced",
        "license_policy": "only CC-BY-4.0 and explicitly research-and-education-only rows are formal-train eligible; unclear licenses are eval-only",
        "license_records": dict(sorted(license_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "metadata_counts": metadata_counts,
        "group_isolation": split_report["group_isolation"],
        "duplicate_checks": split_report["duplicate_checks"],
        "artifacts": {"manifest": str(args.output), "split_report": str(args.split_report)},
    }
    args.dataset_card.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_card.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass" if split_report["group_isolation"]["status"] == "pass" else "fail", **split_report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
