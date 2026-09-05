#!/usr/bin/env python3
"""Mine difficult vehicle-attribute crops without changing labels or splits.

The frozen 60-second acceptance video is intentionally not an input to this
tool. Mining is deterministic and fail-closed: it only adds quality metadata
and never invents a body or colour label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def bool_value(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def image_quality(path: Path) -> tuple[float, float, float]:
    """Return luminance mean, contrast and gradient sharpness proxies."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((96, 96))
        smooth = gray.filter(ImageFilter.GaussianBlur(radius=0.5))
        values = list(smooth.getdata())
        mean = sum(values) / max(1, len(values))
        variance = sum((p - mean) ** 2 for p in values) / max(1, len(values))
        contrast = math.sqrt(variance)
        gradient = 0.0
        count = 0
        for y in range(96):
            for x in range(96):
                if x:
                    gradient += abs(values[y * 96 + x] - values[y * 96 + x - 1])
                    count += 1
                if y:
                    gradient += abs(values[y * 96 + x] - values[(y - 1) * 96 + x])
                    count += 1
        return mean, contrast, gradient / max(1, count)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for field in ("hard_mining_tags", "hard_score", "photometric_mean", "photometric_contrast", "photometric_sharpness"):
        if field not in fieldnames:
            fieldnames.append(field)

    counts = Counter()
    score_bins = Counter()
    missing = 0
    for row in rows:
        image_path = (args.input.parent / row.get("image_path", "")).resolve()
        tags: list[str] = []
        score = 0.0
        lighting = str(row.get("lighting", "")).lower()
        size = str(row.get("vehicle_size", "")).lower()
        quality = str(row.get("crop_quality", "")).lower()
        if lighting in {"night", "low_light"} or bool_value(row.get("night")):
            tags.append("low_light")
            score += 2.0 if lighting == "night" else 1.5
        if size == "small":
            tags.append("small_target")
            score += 2.0
        if str(row.get("occlusion_level", "")).lower() in {"occluded", "truncated"} or bool_value(row.get("occluded")) or bool_value(row.get("truncated")):
            tags.append("occlusion")
            score += 3.0
        if bool_value(row.get("blur")):
            tags.append("motion_blur")
            score += 2.0
        if quality in {"usable", "poor"}:
            tags.append("crop_quality")
            score += 1.0
        if row.get("body_type", "") == "unknown" or row.get("color", "") == "unknown":
            tags.append("unknown_head")
            score += 0.25
        mean = contrast = sharpness = 0.0
        if image_path.exists():
            try:
                mean, contrast, sharpness = image_quality(image_path)
                if mean < 55:
                    tags.append("very_dark")
                    score += 2.0
                elif mean < 85:
                    tags.append("dark")
                    score += 1.0
                if contrast < 22:
                    tags.append("low_contrast")
                    score += 1.0
                if sharpness < 6:
                    tags.append("soft")
                    score += 1.0
            except Exception:
                missing += 1
        else:
            missing += 1
        row["hard_mining_tags"] = ";".join(sorted(set(tags))) or "normal"
        row["hard_score"] = f"{score:.4f}"
        row["photometric_mean"] = f"{mean:.4f}"
        row["photometric_contrast"] = f"{contrast:.4f}"
        row["photometric_sharpness"] = f"{sharpness:.4f}"
        for tag in set(tags):
            counts[tag] += 1
        score_bins[str(min(8, int(score)))] += 1

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "1.0",
        "input_manifest": str(args.input),
        "output_manifest": str(args.output),
        "rows": len(rows),
        "missing_or_unreadable_images": missing,
        "tag_counts": dict(sorted(counts.items())),
        "score_bins": dict(sorted(score_bins.items(), key=lambda item: int(item[0]))),
        "frozen_video_used": False,
        "label_policy": "metadata_only; no labels changed; unknown remains unknown",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
