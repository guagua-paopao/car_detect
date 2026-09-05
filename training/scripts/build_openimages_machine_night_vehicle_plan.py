#!/usr/bin/env python3
"""Build a train-only Open Images vehicle plan from high-confidence machine Night labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


NIGHT_LABEL = "/m/01d74z"
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
    label_source = parser.add_mutually_exclusive_group(required=True)
    label_source.add_argument("--machine-labels", type=Path)
    label_source.add_argument("--night-index", type=Path)
    parser.add_argument("--night-index-report", type=Path)
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--exclude-card", action="append", type=Path, default=[])
    parser.add_argument("--exclude-plan", action="append", type=Path, default=[])
    parser.add_argument("--minimum-machine-confidence", type=float, default=0.95)
    parser.add_argument("--small-area-ratio", type=float, default=0.01)
    parser.add_argument("--minimum-unique-images", type=int, default=1000)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite machine-night source-plan evidence")
    if not 0.5 <= args.minimum_machine_confidence <= 1.0:
        raise ValueError("minimum machine confidence must be in [0.5, 1.0]")

    state = json.loads(args.download_state.read_text(encoding="utf-8"))
    if state.get("status") != "complete_checksum_verified":
        raise RuntimeError("machine image-label download is not checksum verified")
    if state.get("md5") != state.get("expected_md5_from_etag"):
        raise RuntimeError("machine image-label MD5 evidence is inconsistent")
    index_evidence = None
    if args.machine_labels is not None:
        if sha256(args.machine_labels) != state.get("sha256"):
            raise RuntimeError("machine image-label SHA256 differs from download evidence")
        label_input = args.machine_labels
    else:
        if args.night_index_report is None:
            raise ValueError("--night-index-report is required with --night-index")
        index_evidence = json.loads(args.night_index_report.read_text(encoding="utf-8"))
        if index_evidence.get("status") != "pass":
            raise RuntimeError("machine Night index did not pass")
        if index_evidence.get("label_mid") != NIGHT_LABEL:
            raise RuntimeError("machine Night index contains the wrong label")
        if index_evidence.get("source_machine_labels_sha256") != state.get("sha256"):
            raise RuntimeError("machine Night index source does not match download state")
        if index_evidence.get("output_index_sha256") != sha256(args.night_index):
            raise RuntimeError("machine Night index SHA256 mismatch")
        if index_evidence.get("policy", {}).get("frozen_video_used") is not False:
            raise RuntimeError("machine Night index does not prove frozen-video isolation")
        label_input = args.night_index
    if state.get("frozen_video_used") is not False:
        raise RuntimeError("download state does not prove frozen-video isolation")

    excluded_ids: set[str] = set()
    exclusion_evidence = []
    for path in args.exclude_card:
        before = len(excluded_ids)
        card = json.loads(path.read_text(encoding="utf-8"))
        for sample in card.get("samples", []):
            sample_id = str(sample.get("sample_id", ""))
            if sample_id.startswith("oi_train_"):
                excluded_ids.add(sample_id.removeprefix("oi_train_"))
        exclusion_evidence.append({
            "kind": "dataset_card",
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "new_image_ids": len(excluded_ids) - before,
        })
    for path in args.exclude_plan:
        before = len(excluded_ids)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            excluded_ids.update(row.get("image_id", "") for row in csv.DictReader(handle))
        excluded_ids.discard("")
        exclusion_evidence.append({
            "kind": "csv_plan",
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "new_image_ids": len(excluded_ids) - before,
        })

    night_confidence: dict[str, float] = {}
    label_rows_scanned = 0
    night_rows_seen = 0
    with label_input.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label_rows_scanned += 1
            if row.get("LabelName") != NIGHT_LABEL:
                continue
            night_rows_seen += 1
            try:
                confidence = float(row.get("Confidence", ""))
            except ValueError:
                continue
            if confidence >= args.minimum_machine_confidence:
                image_id = row.get("ImageID", "")
                if image_id:
                    night_confidence[image_id] = max(confidence, night_confidence.get(image_id, 0.0))

    detection_rows_scanned = 0
    selected = []
    rejected = Counter()
    candidate_images = set()
    with args.detections.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            detection_rows_scanned += 1
            image_id = row.get("ImageID", "")
            if image_id not in night_confidence:
                continue
            vehicle_class = VEHICLE_LABELS.get(row.get("LabelName", ""))
            if not vehicle_class:
                continue
            candidate_images.add(image_id)
            if image_id in excluded_ids:
                rejected["excluded_existing_source_image"] += 1
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
                "weather_labels": "night_machine_candidate",
                "night_machine_confidence": f"{night_confidence[image_id]:.8f}",
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

    selected_ids = {row["image_id"] for row in selected}
    metadata = {}
    metadata_rows_scanned = 0
    with args.image_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_rows_scanned += 1
            image_id = row.get("ImageID", "")
            if image_id in selected_ids:
                metadata[image_id] = row

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
    fields = list(output_rows[0]) if output_rows else [
        "image_id", "download_ref", "weather_labels", "night_machine_confidence",
        "vehicle_class", "xmin", "xmax", "ymin", "ymax", "bbox_area_ratio",
    ]
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    unique_images = {row["image_id"] for row in output_rows}
    vehicle_counts = Counter(row["vehicle_class"] for row in output_rows)
    small = sum(row["small_target_proxy"] == "true" for row in output_rows)
    occluded = sum(row["occluded"] == "true" for row in output_rows)
    truncated = sum(row["truncated"] == "true" for row in output_rows)
    estimated_bytes = sum(int(metadata[image_id].get("OriginalSize") or 0) for image_id in unique_images)
    gates = {
        "minimum_unique_images": len(unique_images) >= args.minimum_unique_images,
        "official_train_vehicle_boxes_only": True,
        "per_image_cc_by_2_license_verified": not metadata_rejections,
        "prior_openimages_sources_excluded": True,
    }
    status = "pass_source_plan_pending_photometric_audit" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-machine-night-vehicle-source-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "machine_labels": state.get("output"),
        "machine_labels_sha256": state.get("sha256"),
        "label_scan_input": str(label_input.resolve()),
        "label_scan_input_sha256": sha256(label_input),
        "night_index_report": (
            str(args.night_index_report.resolve()) if args.night_index_report else None
        ),
        "night_index_report_sha256": (
            sha256(args.night_index_report) if args.night_index_report else None
        ),
        "download_state": str(args.download_state.resolve()),
        "download_state_sha256": sha256(args.download_state),
        "detections": str(args.detections.resolve()),
        "detections_sha256": sha256(args.detections),
        "image_metadata": str(args.image_metadata.resolve()),
        "image_metadata_sha256": sha256(args.image_metadata),
        "exclusion_evidence": exclusion_evidence,
        "night_label_mid": NIGHT_LABEL,
        "minimum_machine_confidence": args.minimum_machine_confidence,
        "machine_label_rows_scanned": label_rows_scanned,
        "machine_night_rows_seen": night_rows_seen,
        "high_confidence_night_images": len(night_confidence),
        "detection_rows_scanned": detection_rows_scanned,
        "candidate_images_with_vehicle_boxes_before_filter": len(candidate_images),
        "detection_rejections": dict(sorted(rejected.items())),
        "metadata_rows_scanned": metadata_rows_scanned,
        "metadata_rejections": dict(sorted(metadata_rejections.items())),
        "planned_unique_images": len(unique_images),
        "planned_vehicle_boxes": len(output_rows),
        "planned_vehicle_class_counts": dict(sorted(vehicle_counts.items())),
        "small_target_proxy_rows": small,
        "occluded_rows": occluded,
        "truncated_rows": truncated,
        "estimated_original_download_bytes": estimated_bytes,
        "gates": gates,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "machine_night_is_candidate_metadata_only": True,
            "photometric_and_multiteacher_audit_required_before_training": True,
            "machine_label_never_used_as_body_or_color_target": True,
            "official_car_bus_truck_boxes_only": True,
            "group_boxes_excluded": True,
            "depictions_excluded": True,
            "train_split_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "build a bounded download plan, then require crop photometry and strict attribute-teacher consensus",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "night_images": len(night_confidence),
        "planned_images": len(unique_images),
        "boxes": len(output_rows),
        "vehicles": dict(sorted(vehicle_counts.items())),
        "estimated_bytes": estimated_bytes,
        "gates": gates,
    }, ensure_ascii=False))
    return 0 if status == "pass_source_plan_pending_photometric_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
