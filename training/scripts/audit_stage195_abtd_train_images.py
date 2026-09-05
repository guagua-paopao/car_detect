#!/usr/bin/env python3
"""Stream ABTD v3 original train images, crop visually verified exact body boxes, and decontaminate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


EXPECTED_IMAGES_BYTES = 1885610631
EXPECTED_IMAGES_SHA256 = "40c57c19488e05ac6cc50b724b493f7a0827a45b85c20dbdd79c02cee120ac28"
# The published YAML calls class 5 VAN, but the Stage195 R1 contact sheet showed
# that its pixels are predominantly rickshaws/carts/bicycles. Fail closed and
# retain only classes whose official semantics agree with visual inspection.
EXACT_BODY = {1: "bus", 2: "truck"}
NUMBERED = re.compile(r"^(.*?)(\d+)$")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


class NearIndex:
    def __init__(self, distance: int) -> None:
        self.distance = distance
        self.bands: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def match(self, value: int) -> int | None:
        candidates: set[tuple[int, int]] = set()
        for band in range(5):
            width, offset = (13 if band < 4 else 12), band * 13
            candidates.update(self.bands.get((band, (value >> offset) & ((1 << width) - 1)), []))
        for existing, index in candidates:
            if (existing ^ value).bit_count() <= self.distance:
                return index
        return None

    def add(self, value: int, index: int) -> None:
        for band in range(5):
            width, offset = (13 if band < 4 else 12), band * 13
            self.bands[(band, (value >> offset) & ((1 << width) - 1))].append((value, index))


def source_group(stem: str, burst_size: int = 5) -> str:
    match = NUMBERED.match(stem)
    if not match:
        return f"abtd:{stem.casefold()}"
    prefix, number = match.group(1).casefold(), int(match.group(2))
    return f"abtd:{prefix}{number // burst_size:06d}"


def safe_members(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
            raise RuntimeError(f"unsafe ZIP member: {info.filename}")
        if info.is_dir() or pure.suffix.lower() != suffix:
            continue
        key = pure.stem.casefold()
        if key in members:
            raise RuntimeError(f"duplicate basename: {pure.stem}")
        members[key] = info.filename
    return members


def load_cross(paths: list[Path], distance: int) -> tuple[NearIndex, list[dict[str, str]], int]:
    index, records, scanned = NearIndex(distance), [], 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("split") or "train").lower() != "train":
                    continue
                scanned += 1
                text = str(row.get("stage177_dhash64") or row.get("stage188_dhash64") or row.get("crop_dhash64") or "")
                if len(text) != 16:
                    continue
                try:
                    value = int(text, 16)
                except ValueError:
                    continue
                body = str(row.get("body_type") or "").lower() if truthy(row.get("body_type_supervised") or row.get("body_supervised")) else ""
                records.append({"body_type": "" if body == "unknown" else body, "path": str(row.get("image_path") or row.get("crop_path") or "")})
                index.add(value, len(records) - 1)
    return index, records, scanned


def make_contact(rows: list[dict[str, object]], output: Path) -> None:
    labels = [(body, scene) for body in ("bus", "truck", "van") for scene in ("day", "night")]
    tile_w, tile_h, columns = 220, 170, 5
    sheet = Image.new("RGB", (tile_w * columns, tile_h * len(labels)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (body, scene) in enumerate(labels):
        candidates = [row for row in rows if row["stage195_train_eligible"] == "true" and row["body_type"] == body and ((row["confirmed_natural_night"] == "true") == (scene == "night"))]
        candidates.sort(key=lambda row: float(row["quality_score"]), reverse=True)
        for column, row in enumerate(candidates[:columns]):
            with Image.open(str(row["crop_path"])) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 30))
            x = column * tile_w + (tile_w - image.width) // 2
            y = row_index * tile_h + 24 + (tile_h - 30 - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((column * tile_w + 4, row_index * tile_h + 4), f"{body}/{scene} {row['source_image_name']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cross-source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cross-distance", type=int, default=2)
    args = parser.parse_args()
    for value in (args.images, args.labels, args.metadata, args.output_root, *args.cross_source_manifest):
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.images.stat().st_size != EXPECTED_IMAGES_BYTES or sha256(args.images) != EXPECTED_IMAGES_SHA256:
        raise RuntimeError("images ZIP integrity mismatch")
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    originals = {Path(row["image"]).stem.casefold(): row for row in metadata_rows if row["split"] == "train" and row["augmented"] == "no"}
    if len(originals) != 3350:
        raise RuntimeError("unexpected original train metadata count")

    args.output_root.mkdir(parents=True)
    crop_root = args.output_root / "crops"; crop_root.mkdir()
    cross_index, cross_records, cross_scanned = load_cross(args.cross_source_manifest, args.cross_distance)
    output_rows: list[dict[str, object]] = []
    original_train_images_opened = 0
    with zipfile.ZipFile(args.images) as image_zip, zipfile.ZipFile(args.labels) as label_zip:
        if image_zip.testzip() or label_zip.testzip():
            raise RuntimeError("ZIP CRC failure")
        image_members = safe_members(image_zip, ".jpg")
        for info in image_zip.infolist():
            pure = PurePosixPath(info.filename)
            if pure.suffix.lower() in {".jpeg", ".png"}:
                key = pure.stem.casefold()
                if key in image_members:
                    raise RuntimeError(f"duplicate image stem: {key}")
                image_members[key] = info.filename
        label_members = safe_members(label_zip, ".txt")
        missing_images = sorted(set(originals) - set(image_members))
        missing_labels = sorted(set(originals) - set(label_members))
        if missing_images or missing_labels:
            raise RuntimeError(f"original train coverage mismatch images={len(missing_images)} labels={len(missing_labels)}")
        for stem, metadata in sorted(originals.items()):
            boxes: list[tuple[int, list[float]]] = []
            for line in label_zip.read(label_members[stem]).decode("utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                if class_id in EXACT_BODY:
                    boxes.append((class_id, [float(value) for value in parts[1:]]))
            if not boxes:
                continue
            payload = image_zip.read(image_members[stem])
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"decode failure: {stem}")
            original_train_images_opened += 1
            height, width = image.shape[:2]
            for box_index, (class_id, (cx, cy, bw, bh)) in enumerate(boxes):
                left, top = (cx - bw / 2) * width, (cy - bh / 2) * height
                right, bottom = (cx + bw / 2) * width, (cy + bh / 2) * height
                box_w, box_h = right - left, bottom - top
                margin_x, margin_y = 0.06 * box_w, 0.06 * box_h
                x1, y1 = max(0, math.floor(left - margin_x)), max(0, math.floor(top - margin_y))
                x2, y2 = min(width, math.ceil(right + margin_x)), min(height, math.ceil(bottom + margin_y))
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean, contrast = float(gray.mean()), float(gray.std())
                sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                quality_ok = crop.shape[1] >= 20 and crop.shape[0] >= 16 and mean >= 3 and contrast >= 5 and sharpness >= 2
                encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not encoded_ok:
                    continue
                safe_name = hashlib.sha1(f"{stem}:{box_index}".encode()).hexdigest()[:20] + ".jpg"
                crop_path = crop_root / safe_name; crop_path.write_bytes(encoded.tobytes())
                crop_hash, crop_dhash = sha256(crop_path), dhash64(crop)
                scale_score = min(1.0, min(crop.shape[1] / 160, crop.shape[0] / 96))
                quality_score = 0.40 * scale_score + 0.20 * min(1.0, contrast / 55) + 0.20 * min(1.0, math.log1p(sharpness) / math.log1p(500)) + 0.20 * min(1.0, (box_w * box_h) / (0.08 * width * height))
                output_rows.append({
                    "split": "train", "source_dataset": "ABTD-v3", "source_doi": "10.17632/2f9jp8bj45.3", "source_license": "CC BY 4.0",
                    "source_image_name": metadata["image"], "source_group": source_group(stem), "source_box_index": box_index,
                    "crop_path": str(crop_path), "body_type": EXACT_BODY[class_id], "body_type_supervised": "true", "color": "unknown", "color_supervised": "false",
                    "weather": metadata["weather"], "time_of_day": metadata["time_of_day"], "illumination": metadata["illumination"], "illumination_source": metadata["illumination_source"],
                    "confirmed_natural_night": str(metadata["time_of_day"] == "Night" and metadata["illumination_source"] == "annotated").lower(),
                    "lowlight": str(metadata["illumination"] == "Low_Light").lower(),
                    "box_width": round(box_w, 4), "box_height": round(box_h, 4), "box_area_ratio": round(box_w * box_h / (width * height), 8),
                    "crop_width": crop.shape[1], "crop_height": crop.shape[0], "crop_mean_luma": round(mean, 6), "crop_contrast": round(contrast, 6), "crop_sharpness": round(sharpness, 6),
                    "crop_sha256": crop_hash, "crop_dhash64": f"{crop_dhash:016x}", "quality_score": round(quality_score, 6),
                    "stage195_internal_duplicate_of": "", "stage195_cross_source_duplicate_of": "", "stage195_label_conflict": "false",
                    "stage195_train_eligible": str(quality_ok).lower(), "stage195_exclusion_reason": "" if quality_ok else "crop_quality_failed",
                })

    internal_index, internal_records = NearIndex(args.cross_distance), []
    for index, row in enumerate(output_rows):
        if row["stage195_train_eligible"] != "true":
            continue
        value = int(str(row["crop_dhash64"]), 16)
        matched = internal_index.match(value)
        if matched is not None:
            other = output_rows[internal_records[matched]]
            if other["body_type"] != row["body_type"]:
                other["stage195_label_conflict"] = row["stage195_label_conflict"] = "true"
                other["stage195_train_eligible"] = row["stage195_train_eligible"] = "false"
                other["stage195_exclusion_reason"] = row["stage195_exclusion_reason"] = "internal_near_duplicate_label_conflict"
            else:
                row["stage195_internal_duplicate_of"] = other["crop_path"]
                row["stage195_train_eligible"] = "false"
                row["stage195_exclusion_reason"] = "internal_near_duplicate"
            continue
        internal_records.append(index); internal_index.add(value, len(internal_records) - 1)

    for row in output_rows:
        if row["stage195_train_eligible"] != "true":
            continue
        matched = cross_index.match(int(str(row["crop_dhash64"]), 16))
        if matched is None:
            continue
        existing = cross_records[matched]
        row["stage195_cross_source_duplicate_of"] = existing["path"] or matched
        row["stage195_train_eligible"] = "false"
        if existing["body_type"] and existing["body_type"] != row["body_type"]:
            row["stage195_label_conflict"] = "true"
            row["stage195_exclusion_reason"] = "cross_source_near_duplicate_label_conflict"
        else:
            row["stage195_exclusion_reason"] = "cross_source_near_duplicate"

    manifest = args.output_root / "stage195-abtd-train-crops.csv"
    fields = list(output_rows[0]) if output_rows else []
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    contact = args.output_root / "stage195-abtd-contact-sheet.jpg"; make_contact(output_rows, contact)
    eligible = [row for row in output_rows if row["stage195_train_eligible"] == "true"]
    class_counts = Counter(str(row["body_type"]) for row in eligible)
    night_counts = Counter(str(row["body_type"]) for row in eligible if row["confirmed_natural_night"] == "true")
    low_counts = Counter(str(row["body_type"]) for row in eligible if row["lowlight"] == "true")
    exclusions = Counter(str(row["stage195_exclusion_reason"]) or "accepted" for row in output_rows)
    report = {
        "schema_version": "stage195-abtd-train-image-audit-v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_component_pending_teacher_audit_no_training" if eligible else "fail_closed",
        "source": {"doi": "10.17632/2f9jp8bj45.3", "license": "CC BY 4.0", "images_sha256": EXPECTED_IMAGES_SHA256},
        "counts": {
            "original_train_images_considered": len(originals), "original_train_images_opened": original_train_images_opened,
            "augmented_train_images_opened": 0, "validation_or_test_images_opened": 0,
            "materialized_exact_body_crops": len(output_rows), "eligible_rows": len(eligible),
            "eligible_body": dict(sorted(class_counts.items())), "eligible_confirmed_natural_night_body": dict(sorted(night_counts.items())),
            "eligible_lowlight_body": dict(sorted(low_counts.items())), "cross_source_rows_scanned": cross_scanned,
            "internal_duplicates_or_conflicts": sum(bool(row["stage195_internal_duplicate_of"]) or row["stage195_label_conflict"] == "true" for row in output_rows),
            "cross_source_duplicates_or_conflicts": sum(bool(row["stage195_cross_source_duplicate_of"]) for row in output_rows),
            "exclusions": dict(sorted(exclusions.items())), "color_truth_rows": 0,
        },
        "gates": {"image_sha_crc_pass": True, "original_only_pass": True, "class5_van_semantics_rejected_after_r1_visual_audit": True, "grouping_complete": True, "cross_source_decontamination_complete": True, "teacher_audit_complete": False, "training_authorized": False},
        "outputs": {"manifest": str(manifest), "manifest_sha256": sha256(manifest), "contact_sheet": str(contact), "contact_sheet_sha256": sha256(contact), "crop_root": str(crop_root)},
        "policy": {"test_accessed": False, "frozen_video_used": False, "production_model_modified": False, "deployment_performed": False},
    }
    report_path = args.output_root / "stage195-abtd-train-image-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
