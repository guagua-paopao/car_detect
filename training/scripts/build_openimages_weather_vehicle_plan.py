#!/usr/bin/env python3
"""Build a train-only Open Images adverse-weather vehicle download plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


WEATHER_LABELS = {
    "/m/01d74z": "night",
    "/m/06mb1": "rain",
    "/m/0g2z8": "fog",
    "/m/06_dn": "snow",
}
VEHICLE_LABELS = {
    "/m/0k4j": "car",
    "/m/01bjv": "bus",
    "/m/07r04": "truck",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy_number(value: str | None) -> bool:
    try:
        return float(str(value or "0")) >= 0.5
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--exclude-card", action="append", type=Path, default=[])
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--small-area-ratio", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage55 source-plan evidence")

    state = json.loads(args.download_state.read_text(encoding="utf-8"))
    if state.get("status") != "complete_checksum_verified":
        raise RuntimeError("human image-label download is not checksum verified")
    if state.get("md5") != state.get("expected_md5_from_etag"):
        raise RuntimeError("human image-label MD5 evidence is inconsistent")
    if sha256(args.human_labels) != state.get("sha256"):
        raise RuntimeError("human image-label SHA256 differs from download evidence")

    excluded_ids = set()
    exclude_evidence = []
    for card_path in args.exclude_card:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        before = len(excluded_ids)
        for sample in card.get("samples", []):
            sample_id = str(sample.get("sample_id", ""))
            if sample_id.startswith("oi_train_"):
                excluded_ids.add(sample_id.removeprefix("oi_train_"))
        exclude_evidence.append({
            "path": str(card_path.resolve()),
            "sha256": sha256(card_path),
            "new_train_image_ids": len(excluded_ids) - before,
        })

    weather_by_image: dict[str, set[str]] = defaultdict(set)
    label_rows_scanned = 0
    positive_weather_rows = 0
    with args.human_labels.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label_rows_scanned += 1
            weather = WEATHER_LABELS.get(row.get("LabelName", ""))
            if weather and truthy_number(row.get("Confidence")):
                weather_by_image[row["ImageID"]].add(weather)
                positive_weather_rows += 1

    detection_rows_scanned = 0
    rejected = Counter()
    selected = []
    candidate_weather_images_with_vehicle = set()
    with args.detections.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            detection_rows_scanned += 1
            image_id = row.get("ImageID", "")
            if image_id not in weather_by_image:
                continue
            vehicle_class = VEHICLE_LABELS.get(row.get("LabelName", ""))
            if not vehicle_class:
                continue
            candidate_weather_images_with_vehicle.add(image_id)
            if image_id in excluded_ids:
                rejected["existing_source_image"] += 1
                continue
            if truthy_number(row.get("IsGroupOf")):
                rejected["group_of_box"] += 1
                continue
            if truthy_number(row.get("IsDepiction")):
                rejected["depiction_box"] += 1
                continue
            try:
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
            except (KeyError, ValueError):
                rejected["invalid_bbox"] += 1
                continue
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
            if area <= 0.0:
                rejected["invalid_bbox"] += 1
                continue
            selected.append({
                "image_id": image_id,
                "download_ref": f"train/{image_id}",
                "weather_labels": ";".join(sorted(weather_by_image[image_id])),
                "vehicle_class": vehicle_class,
                "xmin": f"{xmin:.8f}",
                "xmax": f"{xmax:.8f}",
                "ymin": f"{ymin:.8f}",
                "ymax": f"{ymax:.8f}",
                "bbox_area_ratio": f"{area:.8f}",
                "small_target_proxy": str(area <= args.small_area_ratio).lower(),
                "occluded": str(truthy_number(row.get("IsOccluded"))).lower(),
                "truncated": str(truthy_number(row.get("IsTruncated"))).lower(),
                "inside": str(truthy_number(row.get("IsInside"))).lower(),
                "source": row.get("Source", ""),
            })

    selected_image_ids = {row["image_id"] for row in selected}
    metadata = {}
    metadata_rows_scanned = 0
    with args.image_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_rows_scanned += 1
            if row.get("ImageID") in selected_image_ids:
                metadata[row["ImageID"]] = row

    output_rows = []
    metadata_rejections = Counter()
    for row in selected:
        source = metadata.get(row["image_id"])
        if source is None:
            metadata_rejections["missing_image_metadata"] += 1
            continue
        license_url = source.get("License", "")
        if "creativecommons.org/licenses/by/2.0" not in license_url:
            metadata_rejections["unexpected_image_license"] += 1
            continue
        output_rows.append({
            **row,
            "original_url": source.get("OriginalURL", ""),
            "landing_url": source.get("OriginalLandingURL", ""),
            "license_url": license_url,
            "author": source.get("Author", ""),
            "author_profile_url": source.get("AuthorProfileURL", ""),
            "title": source.get("Title", ""),
            "original_size_bytes": source.get("OriginalSize", ""),
            "original_md5_base64": source.get("OriginalMD5", ""),
            "rotation": source.get("Rotation", ""),
        })

    output_rows.sort(key=lambda row: (
        row["image_id"], row["vehicle_class"], row["xmin"], row["ymin"]
    ))
    fields = list(output_rows[0]) if output_rows else []
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    by_weather_rows = Counter()
    by_weather_images: dict[str, set[str]] = defaultdict(set)
    by_vehicle = Counter()
    for row in output_rows:
        by_vehicle[row["vehicle_class"]] += 1
        for weather in row["weather_labels"].split(";"):
            by_weather_rows[weather] += 1
            by_weather_images[weather].add(row["image_id"])
    unique_images = {row["image_id"] for row in output_rows}
    small_rows = sum(row["small_target_proxy"] == "true" for row in output_rows)
    occluded_rows = sum(row["occluded"] == "true" for row in output_rows)
    truncated_rows = sum(row["truncated"] == "true" for row in output_rows)
    hard_rows = sum(
        row["small_target_proxy"] == "true"
        or row["occluded"] == "true"
        or row["truncated"] == "true"
        for row in output_rows
    )
    estimated_bytes = sum(
        int(metadata[image_id].get("OriginalSize") or 0)
        for image_id in unique_images
    )
    status = "pass" if len(unique_images) >= 500 and output_rows else "fail"
    report = {
        "schema_version": "openimages-weather-vehicle-source-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "human_labels": str(args.human_labels.resolve()),
        "human_labels_sha256": sha256(args.human_labels),
        "detections": str(args.detections.resolve()),
        "detections_sha256": sha256(args.detections),
        "image_metadata": str(args.image_metadata.resolve()),
        "image_metadata_sha256": sha256(args.image_metadata),
        "download_state": str(args.download_state.resolve()),
        "download_state_sha256": sha256(args.download_state),
        "exclude_cards": exclude_evidence,
        "weather_labels": WEATHER_LABELS,
        "vehicle_labels": VEHICLE_LABELS,
        "label_rows_scanned": label_rows_scanned,
        "positive_weather_label_rows": positive_weather_rows,
        "unique_positive_weather_images": len(weather_by_image),
        "detection_rows_scanned": detection_rows_scanned,
        "weather_images_with_target_vehicle_boxes_before_filter": len(candidate_weather_images_with_vehicle),
        "detection_rejections": dict(sorted(rejected.items())),
        "metadata_rows_scanned": metadata_rows_scanned,
        "metadata_rejections": dict(sorted(metadata_rejections.items())),
        "planned_unique_images": len(unique_images),
        "planned_vehicle_boxes": len(output_rows),
        "planned_vehicle_class_counts": dict(sorted(by_vehicle.items())),
        "planned_weather_box_counts": dict(sorted(by_weather_rows.items())),
        "planned_weather_image_counts": {
            key: len(value) for key, value in sorted(by_weather_images.items())
        },
        "small_target_proxy_area_threshold": args.small_area_ratio,
        "small_target_proxy_rows": small_rows,
        "occluded_rows": occluded_rows,
        "truncated_rows": truncated_rows,
        "hard_condition_union_rows": hard_rows,
        "estimated_original_download_bytes": estimated_bytes,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "human_positive_weather_labels_only": True,
            "official_car_bus_truck_boxes_only": True,
            "group_boxes_excluded": True,
            "depictions_excluded": True,
            "per_image_cc_by_2_license_verified": True,
            "existing_openimages_source_groups_excluded": True,
            "train_split_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "review quotas and estimated bytes before downloading the planned train images",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "images": len(unique_images),
        "boxes": len(output_rows),
        "weather_images": report["planned_weather_image_counts"],
        "vehicles": report["planned_vehicle_class_counts"],
        "small": small_rows,
        "occluded": occluded_rows,
        "truncated": truncated_rows,
        "estimated_bytes": estimated_bytes,
    }, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
