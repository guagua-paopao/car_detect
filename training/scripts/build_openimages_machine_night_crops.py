#!/usr/bin/env python3
"""Build photometrically verified train-only crops from a machine-Night Open Images plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def dhash64(image: Image.Image) -> str:
    reduced = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(reduced.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def photometrics(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    scale = min(1.0, 256.0 / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    border_y = max(2, round(height * 0.12))
    border_x = max(2, round(width * 0.12))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True
    mean = float(gray.mean())
    median = float(np.median(gray))
    p90 = float(np.percentile(gray, 90))
    border_mean = float(gray[mask].mean())
    dark_fraction = float((gray < 55.0).mean())
    highlight_fraction = float((gray > 235.0).mean())
    score = (
        0.28 * float(np.clip((95.0 - mean) / 65.0, 0.0, 1.0))
        + 0.22 * float(np.clip((90.0 - median) / 65.0, 0.0, 1.0))
        + 0.25 * float(np.clip((100.0 - border_mean) / 70.0, 0.0, 1.0))
        + 0.15 * min(1.0, dark_fraction / 0.60)
        + 0.10 * float(np.clip((165.0 - p90) / 105.0, 0.0, 1.0))
    )
    return {
        "mean": mean,
        "median": median,
        "p90": p90,
        "border_mean": border_mean,
        "dark_fraction": dark_fraction,
        "highlight_fraction": highlight_fraction,
        "contrast": float(gray.std()),
        "blur_laplacian": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "lowlight_score": score,
    }


def row_priority(row: dict[str, str]) -> tuple:
    hard = sum(truthy(row.get(key)) for key in ("small_target_proxy", "occluded", "truncated"))
    return (
        -hard,
        -float(row["night_machine_confidence"]),
        float(row["bbox_area_ratio"]),
        row["vehicle_class"],
        row["xmin"],
    )


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
    parser.add_argument("--minimum-source-images", type=int, default=300)
    parser.add_argument("--scene-score-min", type=float, default=0.45)
    parser.add_argument("--scene-mean-max", type=float, default=105.0)
    parser.add_argument("--scene-median-max", type=float, default=100.0)
    parser.add_argument("--scene-border-mean-max", type=float, default=115.0)
    parser.add_argument("--scene-dark-fraction-min", type=float, default=0.18)
    parser.add_argument("--crop-score-min", type=float, default=0.20)
    parser.add_argument("--crop-mean-max", type=float, default=145.0)
    parser.add_argument("--crop-median-max", type=float, default=140.0)
    parser.add_argument("--crop-dark-fraction-min", type=float, default=0.08)
    args = parser.parse_args()
    for path in (args.output_root, args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite machine-night crop evidence: {path}")

    plan_report = json.loads(args.plan_report.read_text(encoding="utf-8"))
    if plan_report.get("status") != "pass_bounded_plan_pending_photometric_audit":
        raise RuntimeError("bounded machine-night plan did not pass")
    plan_hash = sha256(args.plan)
    if plan_report.get("output_plan_sha256") != plan_hash:
        raise RuntimeError("bounded machine-night plan hash mismatch")
    download_state = json.loads(args.download_state.read_text(encoding="utf-8"))
    if download_state.get("status") != "complete_checksum_verified":
        raise RuntimeError("machine-night image download is not checksum verified")
    if download_state.get("plan_sha256") != plan_hash:
        raise RuntimeError("download state does not match machine-night plan")
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
    source_metrics: dict[str, dict[str, float]] = {}
    source_sha: dict[str, str] = {}
    accepted_sources = set()
    class_counts = Counter()
    condition_counts = Counter()
    for index, row in enumerate(selected):
        source_path = (download_root / "images" / f"{row['image_id']}.jpg").resolve()
        source_path.relative_to(download_root)
        if not source_path.is_file():
            rejected["missing_source_image"] += 1
            continue
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if row["image_id"] not in source_metrics:
            source_metrics[row["image_id"]] = photometrics(image)
        metrics = source_metrics[row["image_id"]]
        scene_ok = (
            metrics["lowlight_score"] >= args.scene_score_min
            and metrics["mean"] <= args.scene_mean_max
            and metrics["median"] <= args.scene_median_max
            and metrics["border_mean"] <= args.scene_border_mean_max
            and metrics["dark_fraction"] >= args.scene_dark_fraction_min
        )
        if not scene_ok:
            rejected["scene_photometric_gate"] += 1
            continue
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
        crop = image.crop((left, top, right, bottom))
        crop_metrics = photometrics(crop)
        crop_ok = (
            crop_metrics["lowlight_score"] >= args.crop_score_min
            and crop_metrics["mean"] <= args.crop_mean_max
            and crop_metrics["median"] <= args.crop_median_max
            and crop_metrics["dark_fraction"] >= args.crop_dark_fraction_min
        )
        if not crop_ok:
            rejected["crop_photometric_gate"] += 1
            continue
        crop_name = f"oi_machine_night_{row['image_id']}_{index:07d}.jpg"
        crop_path = crop_root / crop_name
        crop.save(crop_path, quality=95, optimize=True)
        relative_path = crop_path.resolve().relative_to(dataset_root).as_posix()
        if row["image_id"] not in source_sha:
            source_sha[row["image_id"]] = sha256(source_path)
        accepted_sources.add(row["image_id"])
        class_counts[row["vehicle_class"]] += 1
        for condition in ("small_target_proxy", "occluded", "truncated"):
            condition_counts[condition] += truthy(row.get(condition))
        output_rows.append({
            "image_path": relative_path,
            "body_type": "unknown",
            "color": "unknown",
            "crop_quality": "usable",
            "viewpoint": "unknown",
            "blur": str(crop_metrics["blur_laplacian"] < 60.0).lower(),
            "occluded": row["occluded"],
            "truncated": row["truncated"],
            "night": "true",
            "low_light": "true",
            "camera_id": f"static_openimages_machine_night_{row['image_id']}",
            "video_id": f"static_openimages_machine_night_{row['image_id']}",
            "track_group": f"static_openimages_machine_night_{row['image_id']}",
            "split": "train",
            "source_frame_id": row["image_id"],
            "review_status": "pending_multiteacher_pseudolabel",
            "body_type_supervised": "false",
            "color_supervised": "false",
            "annotation_source": "openimages_machine_night_plus_pixel_photometric_audit",
            "source_dataset": "Open-Images-V7",
            "source_manifest": str(args.plan.resolve()),
            "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
            "license_train_eligible": "true",
            "review_method": "machine_night_candidate+scene_and_crop_photometry+pending_attribute_consensus",
            "review_score": "0",
            "formal_train_eligible": "false",
            "pseudo_label": "true",
            "pseudo_label_confidence": "",
            "vehicle_size": "small" if truthy(row["small_target_proxy"]) else "medium_or_large",
            "lighting": "night_machine_photometric_consensus",
            "weather": "night",
            "label_confidence": "unknown",
            "official_vehicle_class": row["vehicle_class"],
            "source_image_id": row["image_id"],
            "night_machine_confidence": row["night_machine_confidence"],
            "source_bbox_area_ratio": row["bbox_area_ratio"],
            "source_box_width_pixels": f"{box_width:.3f}",
            "source_box_height_pixels": f"{box_height:.3f}",
            "source_image_sha256": source_sha[row["image_id"]],
            "source_original_url": row.get("original_url", ""),
            "source_landing_url": row.get("landing_url", ""),
            "source_author": row.get("author", ""),
            "source_license_url": row.get("license_url", ""),
            "crop_width": crop.width,
            "crop_height": crop.height,
            "crop_sha256": sha256(crop_path),
            "dhash64": dhash64(crop),
            "scene_lowlight_score": f"{metrics['lowlight_score']:.6f}",
            "scene_mean_luma": f"{metrics['mean']:.6f}",
            "scene_median_luma": f"{metrics['median']:.6f}",
            "scene_border_mean_luma": f"{metrics['border_mean']:.6f}",
            "scene_dark_fraction": f"{metrics['dark_fraction']:.6f}",
            "crop_lowlight_score": f"{crop_metrics['lowlight_score']:.6f}",
            "crop_mean_luma": f"{crop_metrics['mean']:.6f}",
            "crop_median_luma": f"{crop_metrics['median']:.6f}",
            "crop_dark_fraction": f"{crop_metrics['dark_fraction']:.6f}",
            "crop_sharpness": f"{crop_metrics['blur_laplacian']:.6f}",
        })

    if not output_rows:
        raise RuntimeError("no machine-night crops passed photometric and geometry gates")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    gates = {
        "minimum_output_crops": len(output_rows) >= args.minimum_output_crops,
        "minimum_source_images": len(accepted_sources) >= args.minimum_source_images,
        "all_output_rows_photometrically_verified_night": True,
        "download_checksum_verified": True,
    }
    status = "pass_photometric_pending_multiteacher_review" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-machine-night-crops-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_plan": str(args.plan.resolve()),
        "input_plan_sha256": plan_hash,
        "input_plan_report": str(args.plan_report.resolve()),
        "input_plan_report_sha256": sha256(args.plan_report),
        "download_state": str(args.download_state.resolve()),
        "download_state_sha256": sha256(args.download_state),
        "planned_rows": len(rows),
        "selected_before_audit": len(selected),
        "output_crops": len(output_rows),
        "unique_source_images": len(accepted_sources),
        "official_vehicle_class_counts": dict(sorted(class_counts.items())),
        "hard_condition_counts": dict(sorted(condition_counts.items())),
        "rejections": dict(sorted(rejected.items())),
        "photometric_thresholds": {
            "scene_score_min": args.scene_score_min,
            "scene_mean_max": args.scene_mean_max,
            "scene_median_max": args.scene_median_max,
            "scene_border_mean_max": args.scene_border_mean_max,
            "scene_dark_fraction_min": args.scene_dark_fraction_min,
            "crop_score_min": args.crop_score_min,
            "crop_mean_max": args.crop_mean_max,
            "crop_median_max": args.crop_median_max,
            "crop_dark_fraction_min": args.crop_dark_fraction_min,
        },
        "gates": gates,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "machine_night_is_not_sufficient_without_pixel_audit": True,
            "scene_and_vehicle_crop_photometry_required": True,
            "attributes_unsupervised_until_multiteacher_consensus": True,
            "static_images_not_represented_as_real_tracks": True,
            "snow_never_counted_as_night": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "run strict body and foreground-color teacher audits; conflicts remain unknown",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "crops": len(output_rows),
        "images": len(accepted_sources),
        "classes": dict(sorted(class_counts.items())),
        "conditions": dict(sorted(condition_counts.items())),
        "rejections": dict(sorted(rejected.items())),
    }, ensure_ascii=False))
    return 0 if status == "pass_photometric_pending_multiteacher_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
