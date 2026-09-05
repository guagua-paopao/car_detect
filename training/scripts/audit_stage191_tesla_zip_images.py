#!/usr/bin/env python3
"""Stream Tesla lighting/color images from ZIP, crop vehicles, and fail-close labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


EXPECTED_ARCHIVE_BYTES = 8841072987
EXPECTED_ARCHIVE_MD5 = "9ce3079d42273f07374bd25344809d2f"
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
EXACT_COLOR = {"Blue": "blue", "Green": "green", "Red": "red", "White": "white", "Yellow": "yellow"}
EXACT_BODY = {"3": "sedan", "S": "sedan", "X": "suv", "Y": "suv"}
NUMBERED_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>\.[^.]+)$")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def dhash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def label_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(key, "") for key in ("color", "lighting", "model", "year_3", "year_s", "year_x", "year_y"))


def build_burst_groups(rows: list[dict[str, str]], maximum_group_size: int = 6) -> list[str]:
    groups: list[str] = []
    group_number = 0
    previous_prefix = ""
    previous_number: int | None = None
    previous_signature: tuple[str, ...] | None = None
    current_size = 0
    for row in rows:
        match = NUMBERED_NAME.match(row["image"])
        prefix = match.group("prefix").casefold() if match else row["image"].casefold()
        number = int(match.group("number")) if match else None
        signature = label_signature(row)
        continues = (
            previous_number is not None and number is not None
            and prefix == previous_prefix and number == previous_number + 1
            and signature == previous_signature and current_size < maximum_group_size
        )
        if not continues:
            group_number += 1
            current_size = 0
        current_size += 1
        groups.append(f"tesla-lighting-burst-{group_number:05d}")
        previous_prefix, previous_number, previous_signature = prefix, number, signature
    return groups


def safe_image_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            raise ValueError(f"unsafe ZIP member: {name}")
        if info.is_dir() or pure.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        key = pure.name.casefold()
        if key in members:
            raise ValueError(f"duplicate image basename: {pure.name}")
        members[key] = name
    return members


def select_primary(result, image_shape: tuple[int, int, int]) -> tuple[dict[str, float | int] | None, str]:
    image_height, image_width = image_shape[:2]
    candidates: list[dict[str, float | int]] = []
    if result.boxes is None:
        return None, "no_vehicle_detection"
    for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        class_number = int(class_id)
        if class_number not in VEHICLE_CLASS_IDS:
            continue
        left, top, right, bottom = [float(value) for value in xyxy]
        width, height = max(0.0, right - left), max(0.0, bottom - top)
        area_ratio = width * height / max(1.0, image_width * image_height)
        score = area_ratio * (0.5 + float(confidence))
        candidates.append({
            "left": left, "top": top, "right": right, "bottom": bottom,
            "width": width, "height": height, "area_ratio": area_ratio,
            "confidence": float(confidence), "class_id": class_number, "score": score,
        })
    if not candidates:
        return None, "no_vehicle_detection"
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    primary = candidates[0]
    if int(primary["class_id"]) != 2:
        return None, "primary_not_coco_car"
    if float(primary["area_ratio"]) < 0.05 or float(primary["width"]) < 96 or float(primary["height"]) < 64:
        return None, "primary_vehicle_too_small"
    if len(candidates) > 1 and float(candidates[1]["score"]) >= 0.45 * float(primary["score"]):
        return None, "multiple_plausible_primary_vehicles"
    return primary, "accepted"


def expanded_crop(image: np.ndarray, box: dict[str, float | int], margin: float = 0.06) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    width, height = float(box["width"]), float(box["height"])
    left = max(0, round(float(box["left"]) - width * margin))
    top = max(0, round(float(box["top"]) - height * margin))
    right = min(image_width, round(float(box["right"]) + width * margin))
    bottom = min(image_height, round(float(box["bottom"]) + height * margin))
    if right <= left or bottom <= top:
        raise ValueError("empty expanded crop")
    return image[top:bottom, left:right]


def resized_crop(crop: np.ndarray, maximum_side: int = 512) -> np.ndarray:
    height, width = crop.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return crop
    return cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def load_cross_source_index(paths: list[Path], distance: int = 2) -> tuple[NearIndex, list[dict[str, str]], int]:
    index = NearIndex(distance)
    records: list[dict[str, str]] = []
    scanned = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("split") or "train").lower() != "train":
                    continue
                scanned += 1
                value_text = str(row.get("stage177_dhash64") or row.get("stage188_dhash64") or row.get("crop_dhash64") or "").strip()
                if len(value_text) != 16:
                    continue
                try:
                    value = int(value_text, 16)
                except ValueError:
                    continue
                body = str(row.get("body_type") or "").strip().lower() if truthy(row.get("body_type_supervised") or row.get("body_supervised")) else ""
                color = str(row.get("color") or "").strip().lower() if truthy(row.get("color_supervised")) else ""
                if body == "unknown": body = ""
                if color == "unknown": color = ""
                records.append({"body_type": body, "color": color, "image_path": str(row.get("image_path") or row.get("source_image") or "")})
                index.add(value, len(records) - 1)
    return index, records, scanned


def make_contact_sheet(rows: list[dict[str, object]], output: Path) -> None:
    labels = [(color, lighting) for color in ("white", "blue", "red", "green", "yellow") for lighting in ("Light", "Medium", "Dark")]
    tile_width, tile_height, columns = 220, 170, 4
    sheet = Image.new("RGB", (tile_width * columns, tile_height * len(labels)), "white")
    draw = ImageDraw.Draw(sheet)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["stage191_train_eligible"] == "true" and row["color_supervised"] == "true":
            grouped[(str(row["color"]), str(row["lighting"]))].append(row)
    for label_index, label in enumerate(labels):
        candidates = sorted(grouped[label], key=lambda row: float(row["quality_score"]), reverse=True)[:columns]
        for column, row in enumerate(candidates):
            with Image.open(str(row["crop_path"])) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_width - 8, tile_height - 32))
            x = column * tile_width + (tile_width - image.width) // 2
            y = label_index * tile_height + 26 + (tile_height - 32 - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((column * tile_width + 4, label_index * tile_height + 4), f"{label[0]}/{label[1]} {row['source_image_name']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def run(args: argparse.Namespace) -> dict[str, object]:
    for value in (args.archive, args.labels, args.output_root, *args.cross_source_manifest):
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES or digest(args.archive, "md5") != EXPECTED_ARCHIVE_MD5:
        raise ValueError("archive size or MD5 mismatch")
    archive_sha256 = digest(args.archive, "sha256")
    with args.labels.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if len(labels) != 3026 or len({row["image"].casefold() for row in labels}) != 3026:
        raise ValueError("expected 3026 unique labels")
    burst_groups = build_burst_groups(labels)
    for row, group in zip(labels, burst_groups):
        row["burst_group"] = group

    args.output_root.mkdir(parents=True, exist_ok=False)
    crop_root = args.output_root / "crops"
    crop_root.mkdir()
    cross_index, cross_records, cross_rows_scanned = load_cross_source_index(args.cross_source_manifest, args.cross_distance)
    model = YOLO(str(args.detector))
    if {model.names[index] for index in VEHICLE_CLASS_IDS} != {"car", "motorcycle", "bus", "truck"}:
        raise ValueError("unexpected detector label contract")

    output_rows: list[dict[str, object]] = []
    failures: list[str] = []
    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        members = safe_image_members(archive)
        missing = sorted(row["image"] for row in labels if row["image"].casefold() not in members)
        extras = sorted(name for name in members if name not in {row["image"].casefold() for row in labels})
        if missing or extras:
            raise ValueError(f"ZIP/label coverage mismatch missing={len(missing)} extras={len(extras)}")
        for offset in range(0, len(labels), args.batch_size):
            batch_rows = labels[offset:offset + args.batch_size]
            images: list[np.ndarray] = []
            decoded_rows: list[dict[str, str]] = []
            for row in batch_rows:
                try:
                    payload = archive.read(members[row["image"].casefold()])
                    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        raise ValueError("decode_failed")
                    images.append(image)
                    decoded_rows.append(row)
                except Exception as error:
                    if len(failures) < 50:
                        failures.append(f"{row['image']}: {type(error).__name__}:{error}")
            if not images:
                continue
            results = model.predict(images, imgsz=args.image_size, conf=args.confidence, iou=0.5, classes=sorted(VEHICLE_CLASS_IDS), device=args.device, verbose=False)
            for source, image, result in zip(decoded_rows, images, results):
                primary, detection_reason = select_primary(result, image.shape)
                body = EXACT_BODY.get(source["model"], "unknown")
                color = EXACT_COLOR.get(source["color"], "unknown")
                body_supervised = body != "unknown"
                color_supervised = color != "unknown"
                base: dict[str, object] = {
                    "split": "train", "source_dataset": "Tesla-Lighting-Color-2026",
                    "source_doi": "10.5281/zenodo.19814157", "source_license": "CC BY 4.0",
                    "source_image_name": source["image"], "source_group": source["burst_group"],
                    "lighting": source["lighting"], "scene_label": "low_light" if source["lighting"] in {"Medium", "Dark"} else "daylight",
                    "scene_origin": "source_explicit_visual_lighting_label_not_natural_night_truth",
                    "confirmed_natural_night": "false", "source_color_label": source["color"],
                    "source_model_label": source["model"], "body_type": body,
                    "body_type_supervised": str(body_supervised).lower(), "color": color,
                    "color_supervised": str(color_supervised).lower(), "crop_path": "",
                    "detection_reason": detection_reason, "detector_class_id": "", "detector_confidence": "",
                    "detector_area_ratio": "", "crop_width": "", "crop_height": "",
                    "crop_mean_luma": "", "crop_contrast": "", "crop_sharpness": "",
                    "crop_sha256": "", "crop_dhash64": "", "quality_score": 0.0,
                    "stage191_exact_duplicate_of": "", "stage191_burst_duplicate_of": "",
                    "stage191_cross_source_duplicate_of": "", "stage191_label_conflict": "false",
                    "stage191_train_eligible": "false", "stage191_exclusion_reason": detection_reason,
                }
                if primary is None:
                    output_rows.append(base)
                    continue
                crop = resized_crop(expanded_crop(image, primary))
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean = float(gray.mean())
                contrast = float(gray.std())
                sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                minimum_quality = crop.shape[1] >= 96 and crop.shape[0] >= 64 and mean >= 8.0 and contrast >= 8.0 and sharpness >= 6.0
                encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not encoded_ok:
                    base["stage191_exclusion_reason"] = "jpeg_encode_failed"
                    output_rows.append(base)
                    continue
                safe_name = hashlib.sha1(source["image"].encode("utf-8")).hexdigest()[:20] + ".jpg"
                crop_path = crop_root / safe_name
                crop_path.write_bytes(encoded.tobytes())
                crop_hash = hashlib.sha256(crop_path.read_bytes()).hexdigest()
                crop_dhash = dhash64(crop)
                quality_score = (
                    0.25 * min(1.0, float(primary["confidence"]))
                    + 0.25 * min(1.0, float(primary["area_ratio"]) / 0.45)
                    + 0.20 * min(1.0, min(crop.shape[1] / 320, crop.shape[0] / 192))
                    + 0.15 * min(1.0, contrast / 55.0)
                    + 0.15 * min(1.0, math.log1p(sharpness) / math.log1p(600.0))
                )
                eligible = minimum_quality and (body_supervised or color_supervised)
                base.update({
                    "crop_path": str(crop_path), "detector_class_id": int(primary["class_id"]),
                    "detector_confidence": round(float(primary["confidence"]), 6),
                    "detector_area_ratio": round(float(primary["area_ratio"]), 6),
                    "crop_width": crop.shape[1], "crop_height": crop.shape[0],
                    "crop_mean_luma": round(mean, 6), "crop_contrast": round(contrast, 6),
                    "crop_sharpness": round(sharpness, 6), "crop_sha256": crop_hash,
                    "crop_dhash64": f"{crop_dhash:016x}", "quality_score": round(quality_score, 6),
                    "stage191_train_eligible": str(eligible).lower(),
                    "stage191_exclusion_reason": "" if eligible else ("no_exact_attribute_truth" if not (body_supervised or color_supervised) else "crop_quality_failed"),
                })
                output_rows.append(base)

    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output_rows):
        if row["stage191_train_eligible"] == "true":
            exact_groups[str(row["crop_sha256"])].append(index)
    for indexes in exact_groups.values():
        indexes.sort(key=lambda index: float(output_rows[index]["quality_score"]), reverse=True)
        representative = indexes[0]
        signatures = {(output_rows[index]["body_type"], output_rows[index]["color"]) for index in indexes}
        if len(signatures) > 1:
            for index in indexes:
                output_rows[index]["stage191_label_conflict"] = "true"
                output_rows[index]["stage191_train_eligible"] = "false"
                output_rows[index]["stage191_exclusion_reason"] = "internal_exact_label_conflict"
            continue
        for duplicate in indexes[1:]:
            output_rows[duplicate]["stage191_exact_duplicate_of"] = output_rows[representative]["source_image_name"]
            output_rows[duplicate]["stage191_train_eligible"] = "false"
            output_rows[duplicate]["stage191_exclusion_reason"] = "internal_exact_duplicate"

    burst_candidates: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output_rows):
        if row["stage191_train_eligible"] == "true":
            burst_candidates[str(row["source_group"])].append(index)
    for indexes in burst_candidates.values():
        indexes.sort(key=lambda index: float(output_rows[index]["quality_score"]), reverse=True)
        representative = indexes[0]
        for duplicate in indexes[1:]:
            output_rows[duplicate]["stage191_burst_duplicate_of"] = output_rows[representative]["source_image_name"]
            output_rows[duplicate]["stage191_train_eligible"] = "false"
            output_rows[duplicate]["stage191_exclusion_reason"] = "same_burst_lower_quality"

    for row in output_rows:
        if row["stage191_train_eligible"] != "true":
            continue
        matched = cross_index.match(int(str(row["crop_dhash64"]), 16))
        if matched is None:
            continue
        existing = cross_records[matched]
        row["stage191_cross_source_duplicate_of"] = existing["image_path"] or matched
        row["stage191_train_eligible"] = "false"
        row["stage191_exclusion_reason"] = "cross_source_perceptual_duplicate"
        body_conflict = existing["body_type"] and row["body_type_supervised"] == "true" and existing["body_type"] != row["body_type"]
        color_conflict = existing["color"] and row["color_supervised"] == "true" and existing["color"] != row["color"]
        if body_conflict or color_conflict:
            row["stage191_label_conflict"] = "true"
            row["stage191_exclusion_reason"] = "cross_source_perceptual_label_conflict"

    manifest_path = args.output_root / "stage191-tesla-lighting-color-manifest.csv"
    fields = list(output_rows[0]) if output_rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    contact_path = args.output_root / "stage191-tesla-lighting-color-contact-sheet.jpg"
    make_contact_sheet(output_rows, contact_path)

    eligible = [row for row in output_rows if row["stage191_train_eligible"] == "true"]
    exact_color = Counter(str(row["color"]) for row in eligible if row["color_supervised"] == "true")
    exact_body = Counter(str(row["body_type"]) for row in eligible if row["body_type_supervised"] == "true")
    color_scene = Counter(str(row["lighting"]) for row in eligible if row["color_supervised"] == "true")
    body_scene = Counter(str(row["lighting"]) for row in eligible if row["body_type_supervised"] == "true")
    exclusions = Counter(str(row["stage191_exclusion_reason"]) or "accepted" for row in output_rows)
    report: dict[str, object] = {
        "stage": "stage191_tesla_zip_image_audit_r1",
        "status": "pass_component_pending_teacher_visibility_audit_no_training" if eligible else "fail_closed",
        "source": {"doi": "10.5281/zenodo.19814157", "license": "CC BY 4.0", "archive_sha256": archive_sha256},
        "inputs": {
            "archive": str(args.archive), "archive_bytes": args.archive.stat().st_size,
            "labels": str(args.labels), "detector": str(args.detector), "detector_sha256": digest(args.detector, "sha256"),
            "cross_source_manifests": [str(path) for path in args.cross_source_manifest],
        },
        "counts": {
            "label_rows": len(labels), "burst_groups": len(set(burst_groups)),
            "decoded_or_reported_rows": len(output_rows), "decode_failures": len(failures),
            "materialized_crops": sum(bool(row["crop_path"]) for row in output_rows),
            "eligible_rows": len(eligible), "eligible_exact_color": dict(sorted(exact_color.items())),
            "eligible_exact_body": dict(sorted(exact_body.items())),
            "eligible_color_by_lighting": dict(sorted(color_scene.items())),
            "eligible_body_by_lighting": dict(sorted(body_scene.items())),
            "confirmed_natural_night_rows": 0, "exclusions": dict(sorted(exclusions.items())),
            "cross_source_rows_scanned": cross_rows_scanned,
            "cross_source_duplicates": sum(bool(row["stage191_cross_source_duplicate_of"]) for row in output_rows),
            "label_conflicts": sum(row["stage191_label_conflict"] == "true" for row in output_rows),
        },
        "gates": {
            "archive_size_md5_crc_sha256_pass": True, "zip_label_coverage_pass": True,
            "all_images_streamed_without_full_extraction": True, "foreground_detector_contract_pass": True,
            "burst_grouping_complete": True, "cross_source_decontamination_complete": True,
            "teacher_attribute_visibility_audit_complete": False, "training_authorized": False,
        },
        "scope": {"validation_or_test_images_opened": 0, "frozen_video_used": False, "production_model_modified": False, "deployment_performed": False},
        "failure_examples": failures,
        "outputs": {"manifest": str(manifest_path), "contact_sheet": str(contact_path), "crop_root": str(crop_root)},
    }
    report_path = args.output_root / "stage191-tesla-zip-image-audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--cross-source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--cross-distance", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    report = run(args)
    if report["status"] == "fail_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
