#!/usr/bin/env python3
"""Audit Open Images train boxes and plan small/occluded/truncated vehicle crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


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


def truthy(value: str | None) -> bool:
    try:
        return float(str(value or "0")) >= 0.5
    except ValueError:
        return False


def load_existing(cards: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    existing: dict[str, dict] = {}
    evidence = []
    for card_path in cards:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        before = len(existing)
        for sample in card.get("samples", []):
            sample_id = str(sample.get("sample_id", ""))
            if sample.get("split") != "train" or not sample_id.startswith("oi_train_"):
                continue
            image_id = sample_id.removeprefix("oi_train_")
            existing.setdefault(image_id, {
                "source_card": str(card_path.resolve()),
                "relative_path": sample.get("relative_path", ""),
                "sample_id": sample_id,
            })
        evidence.append({
            "path": str(card_path.resolve()),
            "sha256": sha256(card_path),
            "new_train_image_ids": len(existing) - before,
        })
    return existing, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--existing-card", action="append", type=Path, default=[])
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--small-area-ratio", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Open Images hard-box evidence")

    existing, existing_evidence = load_existing(args.existing_card)
    rows_scanned = 0
    target_rows = 0
    rejected = Counter()
    hard_rows = []
    with args.detections.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_scanned += 1
            vehicle_class = VEHICLE_LABELS.get(row.get("LabelName", ""))
            if not vehicle_class:
                continue
            target_rows += 1
            if truthy(row.get("IsGroupOf")):
                rejected["group_of_box"] += 1
                continue
            if truthy(row.get("IsDepiction")):
                rejected["depiction_box"] += 1
                continue
            try:
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
            except (KeyError, ValueError):
                rejected["invalid_bbox"] += 1
                continue
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
            if area <= 0:
                rejected["invalid_bbox"] += 1
                continue
            small = area <= args.small_area_ratio
            occluded = truthy(row.get("IsOccluded"))
            truncated = truthy(row.get("IsTruncated"))
            if not (small or occluded or truncated):
                rejected["not_hard_condition"] += 1
                continue
            image_id = row["ImageID"]
            present = existing.get(image_id)
            hard_rows.append({
                "image_id": image_id,
                "availability": "existing" if present else "new_download",
                "existing_sample_id": present["sample_id"] if present else "",
                "existing_source_card": present["source_card"] if present else "",
                "existing_relative_path": present["relative_path"] if present else "",
                "vehicle_class": vehicle_class,
                "xmin": f"{xmin:.8f}",
                "xmax": f"{xmax:.8f}",
                "ymin": f"{ymin:.8f}",
                "ymax": f"{ymax:.8f}",
                "bbox_area_ratio": f"{area:.8f}",
                "small_target_proxy": str(small).lower(),
                "occluded": str(occluded).lower(),
                "truncated": str(truncated).lower(),
                "inside": str(truthy(row.get("IsInside"))).lower(),
                "source": row.get("Source", ""),
            })

    hard_image_ids = {row["image_id"] for row in hard_rows}
    metadata = {}
    metadata_rows_scanned = 0
    with args.image_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_rows_scanned += 1
            if row.get("ImageID") in hard_image_ids:
                metadata[row["ImageID"]] = row

    output_rows = []
    metadata_rejections = Counter()
    for row in hard_rows:
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
        row["availability"], row["image_id"], row["vehicle_class"], row["xmin"], row["ymin"]
    ))
    if not output_rows:
        raise RuntimeError("no licensed hard vehicle boxes found")

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    by_availability = Counter(row["availability"] for row in output_rows)
    by_vehicle = Counter(row["vehicle_class"] for row in output_rows)
    by_condition = Counter()
    image_sets: dict[str, set[str]] = {
        "all": set(), "existing": set(), "new_download": set()
    }
    for row in output_rows:
        image_sets["all"].add(row["image_id"])
        image_sets[row["availability"]].add(row["image_id"])
        for key in ("small_target_proxy", "occluded", "truncated"):
            if row[key] == "true":
                by_condition[key] += 1
    new_ids = image_sets["new_download"]
    estimated_new_bytes = sum(
        int(metadata[image_id].get("OriginalSize") or 0) for image_id in new_ids
    )
    report = {
        "schema_version": "openimages-hard-box-source-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if output_rows else "fail",
        "detections": str(args.detections.resolve()),
        "detections_sha256": sha256(args.detections),
        "image_metadata": str(args.image_metadata.resolve()),
        "image_metadata_sha256": sha256(args.image_metadata),
        "existing_cards": existing_evidence,
        "detection_rows_scanned": rows_scanned,
        "target_vehicle_rows": target_rows,
        "detection_rejections": dict(sorted(rejected.items())),
        "metadata_rows_scanned": metadata_rows_scanned,
        "metadata_rejections": dict(sorted(metadata_rejections.items())),
        "planned_vehicle_boxes": len(output_rows),
        "planned_unique_images": len(image_sets["all"]),
        "existing_unique_images": len(image_sets["existing"]),
        "new_download_unique_images": len(new_ids),
        "box_counts_by_availability": dict(sorted(by_availability.items())),
        "vehicle_class_counts": dict(sorted(by_vehicle.items())),
        "hard_condition_counts": dict(sorted(by_condition.items())),
        "small_area_ratio": args.small_area_ratio,
        "estimated_new_download_bytes": estimated_new_bytes,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "official_train_car_bus_truck_boxes_only": True,
            "small_or_occluded_or_truncated_only": True,
            "group_boxes_excluded": True,
            "depictions_excluded": True,
            "per_image_cc_by_2_license_verified": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "review existing-vs-new quotas, visual quality and source bias before crop extraction or image download"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "boxes": len(output_rows),
        "images": len(image_sets["all"]),
        "existing_images": len(image_sets["existing"]),
        "new_images": len(new_ids),
        "conditions": dict(sorted(by_condition.items())),
        "vehicles": dict(sorted(by_vehicle.items())),
        "estimated_new_bytes": estimated_new_bytes
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
