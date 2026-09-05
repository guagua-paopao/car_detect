#!/usr/bin/env python3
"""Create VFG-7 evaluation-only vehicle crops and quality/track metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def quality(image: Image.Image) -> tuple[float, float, float]:
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    contrast = math.sqrt(float(stats.var[0]))
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = float(ImageStat.Stat(edge).var[0])
    return mean, contrast, edge_variance


def frame_number(image_name: str) -> int:
    match = re.search(r"_([0-9]+)$", Path(image_name).stem)
    return int(match.group(1)) if match else 0


def consistent_track_truth(rows: list[dict], label_key: str, supervised_key: str) -> dict[str, str]:
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["track_key"] and parse_bool(row[supervised_key]) and row[label_key] != "unknown":
            values[row["track_key"]].append(row[label_key])
    return {
        track: labels[0]
        for track, labels in values.items()
        if len(labels) >= 3 and len(set(labels)) == 1
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.06)
    parser.add_argument("--min-side", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "attribute_manifest.csv"
    report_path = args.output_root / "build-report.json"

    with args.crop_plan.open("r", encoding="utf-8-sig", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    if any(parse_bool(row.get("training_eligible", "")) for row in plan_rows):
        raise RuntimeError("VFG-7 plan unexpectedly contains training-eligible rows")
    if any(row.get("evaluation_split") not in {"validation", "test"} for row in plan_rows):
        raise RuntimeError("unexpected evaluation split")

    # Verify each source image once before any crop is accepted.
    images: dict[str, dict] = {}
    for row in plan_rows:
        path = Path(row["image_local_path"])
        metadata = (int(row["image_size"]), row["image_sha256"])
        if str(path) in images and images[str(path)]["metadata"] != metadata:
            raise RuntimeError(f"conflicting source metadata: {path}")
        images[str(path)] = {"path": path, "metadata": metadata}
    verified_images = 0
    verified_bytes = 0
    for item in images.values():
        path = item["path"]
        expected_size, expected_sha = item["metadata"]
        if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_sha:
            raise RuntimeError(f"source image verification failed: {path}")
        verified_images += 1
        verified_bytes += expected_size

    body_track_truth = consistent_track_truth(plan_rows, "body_label", "body_supervised")
    color_track_truth = consistent_track_truth(plan_rows, "color_label", "color_supervised")
    track_lengths = Counter(row["track_key"] for row in plan_rows if row["track_key"])
    accepted: list[dict] = []
    rejected = Counter()
    current_source_key = ""
    current_source: Image.Image | None = None
    current_source_mean = 0.0
    try:
        for row in plan_rows:
            source_path = Path(row["image_local_path"])
            key = str(source_path)
            if key != current_source_key:
                if current_source is not None:
                    current_source.close()
                current_source = Image.open(source_path).convert("RGB")
                current_source_mean = quality(current_source.resize((192, 108)))[0]
                current_source_key = key
            source = current_source
            source_mean = current_source_mean
            image_width, image_height = source.size
            x_center = float(row["x_center"]) * image_width
            y_center = float(row["y_center"]) * image_height
            box_width = float(row["width"]) * image_width
            box_height = float(row["height"]) * image_height
            if box_width <= 0 or box_height <= 0:
                rejected["invalid_bbox"] += 1
                continue
            pad_x, pad_y = box_width * args.padding, box_height * args.padding
            left = max(0, math.floor(x_center - box_width / 2 - pad_x))
            top = max(0, math.floor(y_center - box_height / 2 - pad_y))
            right = min(image_width, math.ceil(x_center + box_width / 2 + pad_x))
            bottom = min(image_height, math.ceil(y_center + box_height / 2 + pad_y))
            crop = source.crop((left, top, right, bottom))
            crop_width, crop_height = crop.size
            if min(crop_width, crop_height) < args.min_side:
                rejected["too_small"] += 1
                continue
            mean, contrast, edge_variance = quality(crop)
            if not all(math.isfinite(value) for value in (mean, contrast, edge_variance)):
                rejected["nonfinite_quality"] += 1
                continue
            if source_mean < 65 and mean < 90:
                lighting = "night"
            elif source_mean < 95 or mean < 70:
                lighting = "low_light"
            else:
                lighting = "daylight"
            if min(crop_width, crop_height) < 48 or crop_width * crop_height < 4096:
                vehicle_size = "small"
            elif crop_width * crop_height < 32768:
                vehicle_size = "medium"
            else:
                vehicle_size = "large"
            blur = edge_variance < 20.0
            crop_quality = "usable" if min(crop_width, crop_height) < 24 or blur else "approved"
            split = row["evaluation_split"]
            output_dir = args.output_root / "crops" / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{Path(row['image_name']).stem}_b{int(row['yolo_row_index']):03d}.jpg"
            output_path = output_dir / output_name
            temporary = output_path.with_suffix(".jpg.tmp")
            crop.save(temporary, format="JPEG", quality=95, optimize=True)
            os.replace(temporary, output_path)

            track_key = row["track_key"]
            track_length = track_lengths.get(track_key, 0) if track_key else 0
            if track_key and track_length >= 3:
                body_truth = body_track_truth.get(track_key, "unknown")
                color_truth = color_track_truth.get(track_key, "unknown")
                window_id = f"vfg7:{track_key}"
            else:
                body_truth = row["body_label"] if parse_bool(row["body_supervised"]) else "unknown"
                color_truth = row["color_label"] if parse_bool(row["color_supervised"]) else "unknown"
                window_id = f"vfg7:{split}:{Path(row['image_name']).stem}:b{int(row['yolo_row_index']):03d}"
            accepted.append({
                "image_path": output_path.relative_to(args.output_root).as_posix(),
                "split": split,
                "review_status": "approved",
                "source_dataset": "VFG-7",
                "source_repo": row["repo_id"],
                "source_revision": row["revision"],
                "source_license": row["license"],
                "training_eligible": "false",
                "usage": "independent_evaluation_only",
                "source_split": row["source_split"],
                "source_video": row["source_video"],
                "source_frame_id": row["image_name"],
                "source_image_sha256": row["image_sha256"],
                "source_yolo_row_index": row["yolo_row_index"],
                "alignment_method": row["alignment_method"],
                "agreement_min": row["agreement_min"],
                "body_type": row["body_label"],
                "color": row["color_label"],
                "body_type_supervised": row["body_supervised"],
                "color_supervised": row["color_supervised"],
                "track_id": row["track_id"],
                "track_key": track_key,
                "track_body_truth": body_truth,
                "track_color_truth": color_truth,
                "window_id": window_id,
                "window_pos": frame_number(row["image_name"]),
                "width": crop_width,
                "height": crop_height,
                "source_box_width": f"{box_width:.3f}",
                "source_box_height": f"{box_height:.3f}",
                "vehicle_size": vehicle_size,
                "lighting": lighting,
                "weather": lighting,
                "night": str(lighting == "night").lower(),
                "low_light": str(lighting in {"night", "low_light"}).lower(),
                "blur": str(blur).lower(),
                "occluded": "false",
                "truncated": "false",
                "occlusion_level": "unknown",
                "crop_quality": crop_quality,
                "frame_mean_luma": f"{source_mean:.4f}",
                "crop_mean_luma": f"{mean:.4f}",
                "crop_contrast": f"{contrast:.4f}",
                "edge_variance": f"{edge_variance:.4f}",
                "crop_sha256": sha256(output_path),
            })
    finally:
        if current_source is not None:
            current_source.close()

    fields = list(accepted[0]) if accepted else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)

    split_counts = Counter(row["split"] for row in accepted)
    lighting_counts = Counter(row["lighting"] for row in accepted)
    size_counts = Counter(row["vehicle_size"] for row in accepted)
    body_counts = Counter(row["body_type"] for row in accepted if parse_bool(row["body_type_supervised"]))
    color_counts = Counter(row["color"] for row in accepted if parse_bool(row["color_supervised"]))
    repeated_color_tracks = len({
        row["track_key"] for row in accepted
        if row["track_key"] in color_track_truth and track_lengths[row["track_key"]] >= 3
    })
    report = {
        "schema_version": "vfg7-evaluation-crops-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not rejected.get("missing_image") else "failed",
        "source": {"dataset": "VFG-7", "license": "CC BY-NC 4.0", "training_eligible": False},
        "policy": {
            "usage": "independent_evaluation_only",
            "original_train_usage": "validation_threshold_selection_only",
            "original_val_usage": "withheld_test_only",
            "frozen_video_used": False,
            "model_predictions_used": False,
            "track_truth": "propagate only when at least 3 agreement-1.0 singleton observations are identical; conflicts become unknown",
            "lighting": "pixel-derived proxy, not source ground truth",
            "occlusion": "unknown because VFG structured attributes do not provide audited per-box occlusion truth",
        },
        "input": {
            "crop_plan": str(args.crop_plan.resolve()),
            "crop_plan_sha256": sha256(args.crop_plan),
            "rows": len(plan_rows),
            "verified_source_images": verified_images,
            "verified_source_bytes": verified_bytes,
        },
        "output": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "accepted_crops": len(accepted),
            "rejected": dict(sorted(rejected.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "lighting_proxy_counts": dict(sorted(lighting_counts.items())),
            "vehicle_size_counts": dict(sorted(size_counts.items())),
            "body_supervised_counts": dict(sorted(body_counts.items())),
            "color_supervised_counts": dict(sorted(color_counts.items())),
            "consistent_color_tracks_with_at_least_3_observations": repeated_color_tracks,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
