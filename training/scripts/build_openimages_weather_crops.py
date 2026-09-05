#!/usr/bin/env python3
"""Build train-only vehicle crops from a checksum-verified bounded weather plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image, ImageOps


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
    return str(value or "").strip().lower() == "true"


def row_priority(row: dict[str, str]) -> tuple:
    hard = sum(truthy(row.get(key)) for key in ("small_target_proxy", "occluded", "truncated"))
    weather = len(set(row.get("weather_labels", "").split(";")) - {""})
    area = float(row["bbox_area_ratio"])
    tie = hashlib.sha256(
        f"{row['image_id']}:{row['vehicle_class']}:{row['xmin']}:{row['ymin']}".encode()
    ).hexdigest()
    return (-hard, -weather, area, tie)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--max-boxes-per-image", type=int, default=3)
    parser.add_argument("--minimum-side-pixels", type=int, default=16)
    parser.add_argument("--minimum-area-pixels", type=int, default=400)
    parser.add_argument("--minimum-output-crops", type=int, default=500)
    parser.add_argument("--minimum-night-crops", type=int, default=10)
    parser.add_argument("--blur-threshold", type=float, default=60.0)
    args = parser.parse_args()
    for path in (args.output_root, args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite weather-crop evidence: {path}")
    if not (0.0 <= args.margin <= 0.5):
        raise ValueError("margin must be between 0 and 0.5")
    if args.minimum_output_crops < 1 or args.minimum_night_crops < 0:
        raise ValueError("crop count gates must be non-negative, with at least one output crop")

    plan_report = json.loads(args.plan_report.read_text(encoding="utf-8"))
    if plan_report.get("status") != "pass_bounded_plan":
        raise RuntimeError("bounded weather plan did not pass")
    plan_hash = sha256(args.plan)
    if plan_report.get("output_plan_sha256") != plan_hash:
        raise RuntimeError("bounded weather plan hash mismatch")
    download_state = json.loads(args.download_state.read_text(encoding="utf-8"))
    if download_state.get("status") != "complete_checksum_verified":
        raise RuntimeError("weather image download is not complete and checksum verified")
    if download_state.get("plan_sha256") != plan_hash:
        raise RuntimeError("download state does not match bounded weather plan")
    if download_state.get("frozen_video_used") is not False:
        raise RuntimeError("download state does not prove frozen-video isolation")

    with args.plan.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    selected = [
        row
        for image_id in sorted(by_image)
        for row in sorted(by_image[image_id], key=row_priority)[: args.max_boxes_per_image]
    ]

    crop_root = args.output_root / "crops" / "train"
    crop_root.mkdir(parents=True, exist_ok=False)
    dataset_root = args.dataset_root.resolve()
    download_root = args.download_root.resolve()
    output_rows = []
    rejected = Counter()
    class_counts = Counter()
    weather_counts = Counter()
    condition_counts = Counter()
    unique_sources = set()
    source_sha_cache: dict[str, str] = {}
    for index, row in enumerate(selected):
        source_path = (download_root / "images" / f"{row['image_id']}.jpg").resolve()
        source_path.relative_to(download_root)
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
        crop_name = f"oi_weather_{row['image_id']}_{index:06d}.jpg"
        crop_path = crop_root / crop_name
        crop.save(crop_path, quality=95, optimize=True)
        gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            rejected["unreadable_saved_crop"] += 1
            crop_path.unlink(missing_ok=True)
            continue
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        weathers = sorted(set(row["weather_labels"].split(";")) - {""})
        is_night = "night" in weathers
        relative_path = crop_path.resolve().relative_to(dataset_root).as_posix()
        class_counts[row["vehicle_class"]] += 1
        for weather in weathers:
            weather_counts[weather] += 1
        for condition in ("small_target_proxy", "occluded", "truncated"):
            if truthy(row.get(condition)):
                condition_counts[condition] += 1
        unique_sources.add(row["image_id"])
        if row["image_id"] not in source_sha_cache:
            source_sha_cache[row["image_id"]] = sha256(source_path)
        output_rows.append({
            "image_path": relative_path,
            "body_type": "unknown",
            "color": "unknown",
            "crop_quality": "usable",
            "viewpoint": "unknown",
            "blur": str(blur_score < args.blur_threshold).lower(),
            "occluded": row["occluded"],
            "truncated": row["truncated"],
            "night": str(is_night).lower(),
            "camera_id": f"static_openimages_weather_{row['image_id']}",
            "video_id": f"static_openimages_weather_{row['image_id']}",
            "track_group": f"static_openimages_weather_{row['image_id']}",
            "split": "train",
            "source_frame_id": row["image_id"],
            "review_status": "pending_multiteacher_pseudolabel",
            "body_type_supervised": "false",
            "color_supervised": "false",
            "annotation_source": "openimages_official_weather_box_unlabeled_attribute_v1",
            "source_dataset": "Open-Images-V7",
            "source_manifest": str(args.plan.resolve()),
            "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
            "license_train_eligible": "true",
            "review_method": "official_weather_and_bbox_geometry_pending_attribute_consensus",
            "review_score": "0",
            "formal_train_eligible": "false",
            "pseudo_label": "true",
            "pseudo_label_confidence": "",
            "vehicle_size": "small" if truthy(row["small_target_proxy"]) else "medium_or_large",
            "lighting": "night" if is_night else "adverse_weather_day",
            "weather": ";".join(weathers),
            "label_confidence": "unknown",
            "official_vehicle_class": row["vehicle_class"],
            "source_image_id": row["image_id"],
            "source_bbox_area_ratio": row["bbox_area_ratio"],
            "source_box_width_pixels": f"{box_width:.3f}",
            "source_box_height_pixels": f"{box_height:.3f}",
            "source_image_sha256": source_sha_cache[row["image_id"]],
            "source_original_url": row.get("original_url", ""),
            "source_landing_url": row.get("landing_url", ""),
            "source_author": row.get("author", ""),
            "source_license_url": row.get("license_url", ""),
            "crop_width": crop.width,
            "crop_height": crop.height,
            "crop_sha256": sha256(crop_path),
            "dhash64": dhash64(crop),
            "blur_metric": "variance_of_laplacian_gray",
            "blur_score": f"{blur_score:.6f}",
            "blur_threshold": f"{args.blur_threshold:.6f}",
        })

    if not output_rows:
        raise RuntimeError("no weather crops passed geometry checks")
    fields = list(output_rows[0])
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    night_rows = weather_counts["night"]
    night_fraction = night_rows / len(output_rows)
    gates = {
        "minimum_output_crops": len(output_rows) >= args.minimum_output_crops,
        "minimum_source_night_crops": night_rows >= args.minimum_night_crops,
        "download_checksum_verified": True,
    }
    status = "pass_geometry_pending_multiteacher_review" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-weather-crops-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_plan": str(args.plan.resolve()),
        "input_plan_sha256": plan_hash,
        "input_plan_report": str(args.plan_report.resolve()),
        "input_plan_report_sha256": sha256(args.plan_report),
        "download_state": str(args.download_state.resolve()),
        "download_state_sha256": sha256(args.download_state),
        "download_status": download_state["status"],
        "planned_rows": len(rows),
        "selected_before_geometry": len(selected),
        "output_crops": len(output_rows),
        "unique_source_images": len(unique_sources),
        "official_vehicle_class_counts": dict(sorted(class_counts.items())),
        "weather_crop_counts": dict(sorted(weather_counts.items())),
        "hard_condition_counts": dict(sorted(condition_counts.items())),
        "night_fraction": night_fraction,
        "global_night_lowlight_quota": {
            "required_fraction": 0.30,
            "scope": "combined hard-domain training pool",
            "status": "deferred_to_existing_train_multiframe_lowlight_overlay_and_final_manifest",
            "source_fraction_reported_but_not_used_as_global_gate": night_fraction,
            "snow_counted_as_night": False,
        },
        "geometry_rejections": dict(sorted(rejected.items())),
        "gates": gates,
        "max_boxes_per_image": args.max_boxes_per_image,
        "margin": args.margin,
        "minimum_side_pixels": args.minimum_side_pixels,
        "minimum_area_pixels": args.minimum_area_pixels,
        "minimum_output_crops": args.minimum_output_crops,
        "minimum_night_crops": args.minimum_night_crops,
        "blur_policy": {
            "metric": "variance_of_laplacian_gray",
            "threshold": args.blur_threshold,
        },
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "official_train_boxes_only": True,
            "positive_human_weather_labels_only": True,
            "official_vehicle_class_not_used_as_body_subtype": True,
            "attributes_unsupervised_until_multiteacher_consensus": True,
            "static_images_not_represented_as_real_tracks": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "global_night_lowlight_quota_deferred": True,
            "snow_never_counted_as_night": True,
        },
        "decision": "run strict body and color teacher audits; rejected or conflicting crops remain unknown",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "crops": len(output_rows),
        "images": len(unique_sources),
        "weather": dict(sorted(weather_counts.items())),
        "classes": dict(sorted(class_counts.items())),
        "conditions": dict(sorted(condition_counts.items())),
        "night_fraction": night_fraction,
        "rejections": dict(sorted(rejected.items())),
    }, ensure_ascii=False))
    return 0 if status == "pass_geometry_pending_multiteacher_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
