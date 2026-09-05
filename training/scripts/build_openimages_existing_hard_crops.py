#!/usr/bin/env python3
"""Build a bounded train-only crop pool from already-downloaded Open Images hard boxes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


CLASS_CAPS = {"car": 12000, "bus": 5000, "truck": 5000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> str:
    reduced = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(reduced.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def truthy(value: str | None) -> bool:
    return str(value or "").lower() == "true"


def priority(row: dict) -> tuple:
    conditions = sum(truthy(row[key]) for key in ("small_target_proxy", "occluded", "truncated"))
    area = float(row["bbox_area_ratio"])
    tie = hashlib.sha256(
        f"{row['image_id']}:{row['vehicle_class']}:{row['xmin']}:{row['ymin']}".encode()
    ).hexdigest()
    return (-conditions, area, tie)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--max-boxes-per-image", type=int, default=3)
    parser.add_argument("--minimum-side-pixels", type=int, default=16)
    parser.add_argument("--minimum-area-pixels", type=int, default=400)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite existing-hard crop evidence")
    if not (0 <= args.margin <= 0.5):
        raise ValueError("margin must be between 0 and 0.5")

    with args.plan.open("r", encoding="utf-8-sig", newline="") as handle:
        plan_rows = [row for row in csv.DictReader(handle) if row["availability"] == "existing"]
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in plan_rows:
        by_image[row["image_id"]].append(row)
    bounded = []
    for image_id in sorted(by_image):
        bounded.extend(sorted(by_image[image_id], key=priority)[: args.max_boxes_per_image])
    selected = []
    class_counts = Counter()
    for row in sorted(bounded, key=lambda item: (item["vehicle_class"], priority(item))):
        vehicle_class = row["vehicle_class"]
        if class_counts[vehicle_class] >= CLASS_CAPS[vehicle_class]:
            continue
        class_counts[vehicle_class] += 1
        selected.append(row)
    selected.sort(key=lambda row: (row["image_id"], row["vehicle_class"], priority(row)))

    crop_root = args.output_root / "crops" / "train"
    crop_root.mkdir(parents=True, exist_ok=False)
    output_rows = []
    rejected = Counter()
    output_class_counts = Counter()
    condition_counts = Counter()
    unique_sources = set()
    for index, row in enumerate(selected):
        card_path = Path(row["existing_source_card"]).resolve()
        source_path = (card_path.parent / row["existing_relative_path"]).resolve()
        source_path.relative_to(card_path.parent)
        if not source_path.is_file():
            rejected["missing_source_image"] += 1
            continue
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        xmin, xmax = float(row["xmin"]), float(row["xmax"])
        ymin, ymax = float(row["ymin"]), float(row["ymax"])
        box_width, box_height = (xmax - xmin) * width, (ymax - ymin) * height
        if (
            box_width < args.minimum_side_pixels
            or box_height < args.minimum_side_pixels
            or box_width * box_height < args.minimum_area_pixels
        ):
            rejected["too_few_source_pixels"] += 1
            continue
        dx, dy = (xmax - xmin) * args.margin, (ymax - ymin) * args.margin
        left = max(0, round((xmin - dx) * width))
        top = max(0, round((ymin - dy) * height))
        right = min(width, round((xmax + dx) * width))
        bottom = min(height, round((ymax + dy) * height))
        if right - left < args.minimum_side_pixels or bottom - top < args.minimum_side_pixels:
            rejected["invalid_expanded_crop"] += 1
            continue
        crop = image.crop((left, top, right, bottom))
        crop_name = f"oi_hard_{row['image_id']}_{index:06d}.jpg"
        crop_path = crop_root / crop_name
        crop.save(crop_path, quality=95, optimize=True)
        relative_path = crop_path.resolve().relative_to(args.dataset_root.resolve()).as_posix()
        output_class_counts[row["vehicle_class"]] += 1
        for condition in ("small_target_proxy", "occluded", "truncated"):
            if truthy(row[condition]):
                condition_counts[condition] += 1
        unique_sources.add(row["image_id"])
        output_rows.append({
            "image_path": relative_path,
            "body_type": "unknown",
            "color": "unknown",
            "crop_quality": "hard_bbox_margin08",
            "viewpoint": "unknown",
            "blur": "unknown",
            "occluded": row["occluded"],
            "truncated": row["truncated"],
            "night": "false",
            "camera_id": f"static_camera_{row['image_id']}",
            "video_id": f"static_image_{row['image_id']}",
            "track_group": f"static_image_{row['image_id']}",
            "split": "train",
            "source_frame_id": row["image_id"],
            "review_status": "pending_multiteacher_pseudolabel",
            "body_type_supervised": "false",
            "color_supervised": "false",
            "annotation_source": "openimages_official_hard_box_unlabeled_attribute_v1",
            "source_dataset": "Open-Images-V7",
            "source_manifest": str(args.plan.resolve()),
            "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
            "license_train_eligible": "true",
            "review_method": "official_bbox_geometry_pending_attribute_consensus",
            "review_score": "0",
            "formal_train_eligible": "false",
            "pseudo_label": "true",
            "pseudo_label_confidence": "",
            "vehicle_size": "small" if truthy(row["small_target_proxy"]) else "medium_or_large",
            "lighting": "unknown",
            "label_confidence": "unknown",
            "official_vehicle_class": row["vehicle_class"],
            "source_image_id": row["image_id"],
            "source_bbox_area_ratio": row["bbox_area_ratio"],
            "source_box_width_pixels": f"{box_width:.3f}",
            "source_box_height_pixels": f"{box_height:.3f}",
            "crop_width": crop.width,
            "crop_height": crop.height,
            "crop_sha256": sha256(crop_path),
            "dhash64": dhash64(crop),
        })

    if not output_rows:
        raise RuntimeError("no hard crops passed geometry checks")
    fields = list(output_rows[0])
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "schema_version": "openimages-existing-hard-crops-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_geometry_pending_multiteacher_review",
        "input_plan": str(args.plan.resolve()),
        "input_plan_sha256": sha256(args.plan),
        "existing_plan_boxes": len(plan_rows),
        "bounded_boxes_before_class_caps": len(bounded),
        "selected_before_geometry": len(selected),
        "output_crops": len(output_rows),
        "unique_source_images": len(unique_sources),
        "official_vehicle_class_counts": dict(sorted(output_class_counts.items())),
        "hard_condition_counts": dict(sorted(condition_counts.items())),
        "geometry_rejections": dict(sorted(rejected.items())),
        "class_caps": CLASS_CAPS,
        "max_boxes_per_image": args.max_boxes_per_image,
        "margin": args.margin,
        "minimum_side_pixels": args.minimum_side_pixels,
        "minimum_area_pixels": args.minimum_area_pixels,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "existing_images_only": True,
            "official_train_boxes_only": True,
            "official_vehicle_class_not_used_as_body_subtype": True,
            "body_and_color_unsupervised_until_consensus": True,
            "static_images_not_represented_as_tracks": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "run diverse multi-teacher multi-view attribute consensus; rejected crops remain unknown and unsupervised"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "crops": len(output_rows),
        "images": len(unique_sources),
        "classes": dict(sorted(output_class_counts.items())),
        "conditions": dict(sorted(condition_counts.items())),
        "rejections": dict(sorted(rejected.items()))
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
