#!/usr/bin/env python3
"""Mine train-only hard vehicle color proposals from the larger Open Images pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


TARGET_CLASSES = {"car", "bus", "truck"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_foreground_function():
    path = Path(__file__).with_name("build_bmd45_track_color_pseudolabels.py")
    spec = importlib.util.spec_from_file_location("stage54_foreground", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load foreground helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.foreground_color_evidence, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-card", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage54 evidence")

    source_card = json.loads(args.source_card.read_text(encoding="utf-8"))
    exclude_card = json.loads(args.exclude_card.read_text(encoding="utf-8"))
    excluded_ids = {sample["sample_id"] for sample in exclude_card.get("samples", [])}
    foreground_color_evidence, helper_path = load_foreground_function()
    samples = [sample for sample in source_card["samples"] if sample.get("split") == "train"]
    if not samples:
        raise RuntimeError("no train samples in the larger Open Images source card")

    crop_root = args.output_root / "crops" / "train"
    crop_root.mkdir(parents=True, exist_ok=False)
    rows = []
    proposal_counts = Counter()
    rejection_reasons = Counter()
    official_counts = Counter()
    low_light_rows = 0
    small_rows = 0
    target_detections = 0
    source_images_opened = 0
    excluded_overlap_images = 0
    invalid_boxes = 0
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        if sample["sample_id"] in excluded_ids:
            excluded_overlap_images += 1
            continue
        detections = sample.get("annotations", {}).get("detections", [])
        selected = [
            (index, detection) for index, detection in enumerate(detections)
            if detection.get("vehicle_class") in TARGET_CLASSES
        ]
        if not selected:
            continue
        source_path = (args.source_root.resolve() / sample["relative_path"]).resolve()
        source_path.relative_to(args.source_root.resolve())
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
        source_images_opened += 1
        width, height = image.size
        for annotation_index, detection in selected:
            target_detections += 1
            x1, y1, x2, y2 = map(float, detection["bbox_xyxy_norm"])
            left = max(0, min(width - 1, int(x1 * width)))
            top = max(0, min(height - 1, int(y1 * height)))
            right = max(left + 1, min(width, int(round(x2 * width))))
            bottom = max(top + 1, min(height, int(round(y2 * height))))
            crop_width, crop_height = right - left, bottom - top
            if crop_width < 18 or crop_height < 18:
                invalid_boxes += 1
                continue
            crop = image.crop((left, top, right, bottom))
            crop_name = f"train_{sample['sample_id']}_{annotation_index:03d}.jpg"
            crop_path = crop_root / crop_name
            crop.save(crop_path, quality=95, subsampling=0)
            evidence = foreground_color_evidence(crop_path)
            proposal = str(evidence.get("label", "unknown"))
            if proposal == "unknown":
                rejection_reasons[str(evidence.get("reason", "unknown"))] += 1
                continue
            area_pixels = crop_width * crop_height
            area_ratio = (crop_width / width) * (crop_height / height)
            is_small = min(crop_width, crop_height) < 48 or area_pixels < 12288 or area_ratio < 0.005
            is_low_light = float(evidence.get("mean_value", 256.0)) < 75.0
            small_rows += int(is_small)
            low_light_rows += int(is_low_light)
            official_class = str(detection["vehicle_class"])
            official_counts[official_class] += 1
            proposal_counts[proposal] += 1
            source_group = sample["sample_id"]
            rows.append({
                "image_path": str(crop_path.relative_to(args.output_root.parent)).replace("\\", "/"),
                "body_type": "unknown",
                "color": proposal,
                "crop_quality": "usable" if is_small else "good",
                "viewpoint": "unknown",
                "blur": "unknown",
                "occluded": str(bool(detection.get("occluded"))).lower(),
                "truncated": str(bool(detection.get("truncated"))).lower(),
                "night": str(is_low_light).lower(),
                "camera_id": f"static_camera_{source_group}",
                "video_id": f"static_image_{source_group}",
                "track_group": f"static_image_{source_group}",
                "split": "train",
                "source_frame_id": source_group,
                "review_status": "pending_teacher_audit",
                "body_type_supervised": "false",
                "color_supervised": "true",
                "annotation_source": "openimages_large_foreground_color_proposal_v1",
                "source_dataset": "Open-Images-V7",
                "source_manifest": str(args.source_card.resolve()),
                "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                "license_train_eligible": "true",
                "color_review_status": "foreground_proposal_pending_teacher_audit",
                "review_method": "foreground_pixels_only_proposal_not_approved_label",
                "review_score": f"{float(evidence['score']):.6f}",
                "formal_train_eligible": "false",
                "pseudo_label": "true",
                "pseudo_label_confidence": "",
                "vehicle_size": "small" if is_small else "medium_or_large",
                "lighting": "low_light_proxy" if is_low_light else "non_low_light_proxy",
                "foreground_score": f"{float(evidence['score']):.6f}",
                "foreground_margin": f"{float(evidence['margin']):.6f}",
                "foreground_mean_value": f"{float(evidence.get('mean_value', 0.0)):.6f}",
                "foreground_mean_saturation": f"{float(evidence.get('mean_saturation', 0.0)):.6f}",
                "official_vehicle_class": official_class,
                "official_annotation_index": str(annotation_index),
                "source_sample_id": sample["sample_id"],
                "source_bbox_area_ratio": f"{area_ratio:.8f}",
                "source_crop_width": str(crop_width),
                "source_crop_height": str(crop_height),
            })

    fields = list(rows[0]) if rows else []
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status = "pass" if len(rows) >= 500 and len(proposal_counts) == 8 else "fail"
    report = {
        "schema_version": "openimages-large-hard-color-proposal-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_card": str(args.source_card.resolve()),
        "source_card_sha256": sha256(args.source_card),
        "exclude_card": str(args.exclude_card.resolve()),
        "exclude_card_sha256": sha256(args.exclude_card),
        "foreground_helper": str(helper_path.resolve()),
        "foreground_helper_sha256": sha256(helper_path),
        "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
        "source_train_images": len(samples),
        "excluded_overlap_source_images": excluded_overlap_images,
        "source_images_opened": source_images_opened,
        "target_detections": target_detections,
        "invalid_or_tiny_boxes": invalid_boxes,
        "foreground_proposal_rows": len(rows),
        "foreground_proposal_color_counts": dict(sorted(proposal_counts.items())),
        "official_vehicle_class_counts": dict(sorted(official_counts.items())),
        "foreground_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "proposal_low_light_proxy_rows": low_light_rows,
        "proposal_small_rows": small_rows,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "source_train_split_only": True,
            "all_existing_pilot_source_ids_excluded": True,
            "official_target_classes_only": sorted(TARGET_CLASSES),
            "proposals_are_not_approved_labels": True,
            "multi_teacher_audit_required": True,
            "validation_or_test_images_opened": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "run fail-closed three-view multi-teacher audit before any row may enter auxiliary training",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "target_detections": target_detections,
        "proposals": len(rows),
        "colors": dict(sorted(proposal_counts.items())),
        "low_light": low_light_rows,
        "small": small_rows,
    }, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
