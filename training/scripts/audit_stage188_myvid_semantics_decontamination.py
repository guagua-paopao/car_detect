#!/usr/bin/env python3
"""Build a conservative, original-frame-only MY-VID v2 component manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NearIndex:
    def __init__(self, distance: int) -> None:
        self.distance = distance
        self.bands: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def match(self, value: int) -> int | None:
        candidates: set[tuple[int, int]] = set()
        for band_index in range(5):
            width = 13 if band_index < 4 else 12
            offset = band_index * 13
            mask = (1 << width) - 1
            candidates.update(self.bands.get((band_index, (value >> offset) & mask), []))
        for existing, index in candidates:
            if (existing ^ value).bit_count() <= self.distance:
                return index
        return None

    def add(self, value: int, index: int) -> None:
        for band_index in range(5):
            width = 13 if band_index < 4 else 12
            offset = band_index * 13
            mask = (1 << width) - 1
            self.bands[(band_index, (value >> offset) & mask)].append((value, index))


def dhash64(image) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def crop_from_row(image, row: dict[str, str]):
    image_height, image_width = image.shape[:2]
    x = float(row["x_center"])
    y = float(row["y_center"])
    width = float(row["width"])
    height = float(row["height"])
    left = max(0, round((x - width / 2) * image_width))
    top = max(0, round((y - height / 2) * image_height))
    right = min(image_width, round((x + width / 2) * image_width))
    bottom = min(image_height, round((y + height / 2) * image_height))
    if right <= left or bottom <= top:
        raise ValueError("empty crop")
    return image[top:bottom, left:right]


def load_semantics(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass_conservative_mapping":
        raise ValueError("semantic approval is not pass_conservative_mapping")
    policy = payload.get("policy", {})
    if not (
        policy.get("validation_images_opened") is False
        and policy.get("test_images_opened") is False
        and policy.get("frozen_video_used") is False
        and policy.get("color_truth_available") is False
        and policy.get("confirmed_per_frame_night_truth_available") is False
    ):
        raise ValueError("semantic policy is unsafe")
    mapping = {int(key): value for key, value in payload["mapping"].items()}
    if set(mapping) != set(range(6)):
        raise ValueError("expected six semantic mappings")
    return mapping


def load_existing_index(path: Path, distance: int) -> tuple[NearIndex, list[dict[str, str]], int]:
    index = NearIndex(distance)
    records: list[dict[str, str]] = []
    train_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split") or "train").lower() != "train":
                continue
            train_rows += 1
            value_text = str(row.get("stage177_dhash64") or row.get("crop_dhash64") or "").strip()
            if len(value_text) != 16:
                continue
            try:
                value = int(value_text, 16)
            except ValueError:
                continue
            body = str(row.get("body_type") or "").strip().lower()
            if not truthy(row.get("body_type_supervised")) or body in {"", "unknown"}:
                body = ""
            records.append({"body_type": body, "image_path": str(row.get("image_path") or "")})
            index.add(value, len(records) - 1)
    return index, records, train_rows


def run(
    stage187_manifest: Path,
    existing_manifest: Path,
    semantics_path: Path,
    output_manifest: Path,
    report_path: Path,
    near_distance: int = 4,
) -> dict[str, object]:
    for path in (stage187_manifest, existing_manifest, semantics_path, output_manifest, report_path):
        if any(marker in str(path).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    mapping = load_semantics(semantics_path)
    existing_index, existing_records, existing_train_rows = load_existing_index(existing_manifest, near_distance)

    rows: list[dict[str, object]] = []
    input_rows = Counter()
    read_errors: list[str] = []
    cached_path = ""
    cached_image = None
    with stage187_manifest.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            input_rows["all"] += 1
            input_rows[f"augmentation_{source.get('export_augmentation', 'unknown')}"] += 1
            if source.get("export_augmentation") != "original":
                continue
            image_path = str(source["source_image"])
            try:
                if image_path != cached_path:
                    encoded = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
                    cached_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    cached_path = image_path
                    if cached_image is None:
                        raise ValueError("decode_failed")
                crop = crop_from_row(cached_image, source)
                crop_height, crop_width = crop.shape[:2]
                dhash = dhash64(crop)
                pixel_sha = hashlib.sha256(
                    f"{crop_width}x{crop_height}x{crop.shape[2]}:".encode("ascii") + crop.tobytes()
                ).hexdigest()
            except Exception as error:
                if len(read_errors) < 50:
                    read_errors.append(f"{image_path}:box{source.get('box_index')}: {type(error).__name__}:{error}")
                continue
            class_id = int(source["numeric_class_id"])
            semantic = mapping[class_id]
            body_supervised = bool(semantic["body_supervised"])
            body_type = str(semantic["body_type"])
            minimum_quality = crop_width >= 16 and crop_height >= 16 and crop_width * crop_height >= 400
            rows.append({
                **source,
                "body_type": body_type,
                "coarse_body_family": str(semantic["coarse_body_family"]),
                "body_supervised": str(body_supervised).lower(),
                "color": "unknown",
                "color_supervised": "false",
                "confirmed_night": "false",
                "lowlight_proxy": str(source.get("scene_label") == "low_light").lower(),
                "stage188_crop_width": crop_width,
                "stage188_crop_height": crop_height,
                "stage188_minimum_quality": str(minimum_quality).lower(),
                "stage188_pixel_sha256": pixel_sha,
                "stage188_dhash64": f"{dhash:016x}",
                "stage188_internal_duplicate_of": "",
                "stage188_internal_label_conflict": "false",
                "stage188_cross_source_duplicate_of": "",
                "stage188_cross_source_label_conflict": "false",
                "stage188_component_train_eligible": str(body_supervised and minimum_quality).lower(),
                "stage188_exclusion_reason": "" if body_supervised and minimum_quality else (
                    "coarse_or_out_of_scope_class" if not body_supervised else "crop_below_16px_or_400px"
                ),
            })

    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        exact_groups[str(row["stage188_pixel_sha256"])].append(index)
    internal_exact_duplicates = internal_conflict_rows = 0
    internal_representatives: list[int] = []
    for indexes in exact_groups.values():
        supervised_labels = {
            str(rows[index]["body_type"]) for index in indexes
            if rows[index]["body_supervised"] == "true"
        }
        if len(supervised_labels) > 1:
            internal_conflict_rows += len(indexes)
            for index in indexes:
                rows[index]["stage188_internal_label_conflict"] = "true"
                rows[index]["stage188_component_train_eligible"] = "false"
                rows[index]["stage188_exclusion_reason"] = "internal_exact_label_conflict"
            continue
        internal_representatives.append(indexes[0])
        for duplicate in indexes[1:]:
            rows[duplicate]["stage188_internal_duplicate_of"] = indexes[0]
            rows[duplicate]["stage188_component_train_eligible"] = "false"
            rows[duplicate]["stage188_exclusion_reason"] = "internal_exact_duplicate"
            internal_exact_duplicates += 1

    by_video: dict[str, list[int]] = defaultdict(list)
    for index in internal_representatives:
        if rows[index]["stage188_internal_label_conflict"] == "false":
            by_video[str(rows[index]["source_video_id"])].append(index)
    internal_near_duplicates = 0
    for indexes in by_video.values():
        near = NearIndex(near_distance)
        for index in indexes:
            value = int(str(rows[index]["stage188_dhash64"]), 16)
            matched = near.match(value)
            if matched is None:
                near.add(value, index)
                continue
            left = str(rows[index]["body_type"]) if rows[index]["body_supervised"] == "true" else ""
            right = str(rows[matched]["body_type"]) if rows[matched]["body_supervised"] == "true" else ""
            if left and right and left != right:
                for conflict in (index, matched):
                    rows[conflict]["stage188_internal_label_conflict"] = "true"
                    rows[conflict]["stage188_component_train_eligible"] = "false"
                    rows[conflict]["stage188_exclusion_reason"] = "internal_near_label_conflict"
                internal_conflict_rows += 2
                continue
            rows[index]["stage188_internal_duplicate_of"] = matched
            rows[index]["stage188_component_train_eligible"] = "false"
            rows[index]["stage188_exclusion_reason"] = "internal_near_duplicate_same_video"
            internal_near_duplicates += 1

    cross_duplicates = cross_conflicts = 0
    for row in rows:
        if row["stage188_component_train_eligible"] != "true":
            continue
        value = int(str(row["stage188_dhash64"]), 16)
        matched = existing_index.match(value)
        if matched is None:
            continue
        row["stage188_cross_source_duplicate_of"] = existing_records[matched]["image_path"] or matched
        row["stage188_component_train_eligible"] = "false"
        row["stage188_exclusion_reason"] = "cross_source_perceptual_duplicate"
        cross_duplicates += 1
        existing_body = existing_records[matched]["body_type"]
        if existing_body and existing_body != row["body_type"]:
            row["stage188_cross_source_label_conflict"] = "true"
            row["stage188_exclusion_reason"] = "cross_source_perceptual_label_conflict"
            cross_conflicts += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    eligible = [row for row in rows if row["stage188_component_train_eligible"] == "true"]
    eligible_body = Counter(str(row["body_type"]) for row in eligible)
    eligible_lowlight = Counter(str(row["body_type"]) for row in eligible if row["lowlight_proxy"] == "true")
    original_class_counts = Counter(str(row["numeric_class_id"]) for row in rows)
    status = "pass_component_train_only_auxiliary_pending_quota_merge" if eligible else "fail_closed"
    report: dict[str, object] = {
        "stage": "stage188_myvid_v2_semantics_decontamination_r1",
        "status": status,
        "inputs": {
            "stage187_manifest": str(stage187_manifest),
            "stage187_manifest_sha256": file_sha256(stage187_manifest),
            "existing_stage177_manifest": str(existing_manifest),
            "existing_stage177_manifest_sha256": file_sha256(existing_manifest),
            "semantics": str(semantics_path),
            "semantics_sha256": file_sha256(semantics_path),
        },
        "counts": {
            "stage187_all_export_rows": input_rows["all"],
            "stage187_original_rows": len(rows),
            "original_numeric_class_rows": dict(sorted(original_class_counts.items())),
            "existing_train_rows_scanned": existing_train_rows,
            "existing_dhash_rows_indexed": len(existing_records),
            "read_errors": len(read_errors),
            "internal_exact_duplicates": internal_exact_duplicates,
            "internal_near_duplicates_same_video": internal_near_duplicates,
            "internal_label_conflict_rows": internal_conflict_rows,
            "cross_source_perceptual_duplicates": cross_duplicates,
            "cross_source_label_conflicts": cross_conflicts,
            "component_train_eligible_rows": len(eligible),
            "component_train_eligible_body": dict(sorted(eligible_body.items())),
            "component_train_eligible_lowlight_proxy_body": dict(sorted(eligible_lowlight.items())),
            "confirmed_night_truth_rows": 0,
            "color_truth_rows": 0,
        },
        "gates": {
            "official_class_semantics_verified": True,
            "original_export_only": True,
            "source_video_grouping_available": True,
            "track_identity_available": False,
            "internal_dedup_complete": True,
            "cross_source_decontamination_complete": True,
            "component_train_only_auxiliary_eligible": bool(eligible),
            "overall_training_authorized": False,
            "reason_overall_training_held": "confirmed night quota remains unmet and this source has no color truth",
        },
        "scope": {
            "validation_images_opened": 0,
            "test_images_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "read_error_examples": read_errors,
        "output_manifest": str(output_manifest),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage187-manifest", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=4)
    args = parser.parse_args()
    report = run(
        args.stage187_manifest,
        args.existing_manifest,
        args.semantics,
        args.output_manifest,
        args.report,
        args.near_distance,
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] == "fail_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
