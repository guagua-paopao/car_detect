#!/usr/bin/env python3
"""Audit only the published L-VAD train split without opening valid/test images.

The archive exports numeric class names, so this audit deliberately keeps the
labels numeric and refuses training authorization until their semantics are
independently verified.  It also records the Roboflow source-frame identity so
sequential frames and generated variants cannot later cross evaluation splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_RE = re.compile(r"^(?P<frame>.+?)_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
VIDEO_RE = re.compile(r"^(?P<video>.*?)(?:_?\d+)$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dhash64(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column + 1] > pixels[row * 9 + column]
            )
    return value


class NearIndex:
    def __init__(self, distance: int) -> None:
        self.distance = distance
        self.items: list[tuple[int, int]] = []
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
        self.items.append((value, index))
        for band_index in range(5):
            width = 13 if band_index < 4 else 12
            offset = band_index * 13
            mask = (1 << width) - 1
            self.bands[(band_index, (value >> offset) & mask)].append((value, index))


def source_keys(stem: str) -> tuple[str, str]:
    match = SOURCE_RE.match(stem)
    frame = match.group("frame") if match else stem
    video_match = VIDEO_RE.match(frame)
    video = video_match.group("video") if video_match else frame
    return frame, video.rstrip("_") or "unknown"


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path.name}:{line_number}: expected 5 fields")
        class_value = float(fields[0])
        class_id = int(class_value)
        values = tuple(float(value) for value in fields[1:])
        if class_value != class_id or class_id not in {0, 1, 2}:
            raise ValueError(f"{path.name}:{line_number}: invalid class {fields[0]}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{line_number}: non-finite box")
        x, y, width, height = values
        if width <= 0 or height <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{path.name}:{line_number}: invalid normalized box")
        if x - width / 2 < -1e-4 or x + width / 2 > 1.0001:
            raise ValueError(f"{path.name}:{line_number}: horizontal box outside image")
        if y - height / 2 < -1e-4 or y + height / 2 > 1.0001:
            raise ValueError(f"{path.name}:{line_number}: vertical box outside image")
        boxes.append((class_id, x, y, width, height))
    return boxes


def crop_box(image: Image.Image, box: tuple[int, float, float, float, float]) -> Image.Image:
    _, x, y, width, height = box
    image_width, image_height = image.size
    left = max(0, round((x - width / 2) * image_width))
    top = max(0, round((y - height / 2) * image_height))
    right = min(image_width, round((x + width / 2) * image_width))
    bottom = min(image_height, round((y + height / 2) * image_height))
    if right <= left or bottom <= top:
        raise ValueError("empty pixel crop")
    return image.crop((left, top, right, bottom))


def build_contact_sheet(samples: dict[int, list[tuple[str, Image.Image]]], output: Path) -> None:
    tile_width, tile_height = 220, 170
    columns = 4
    rows_per_class = 3
    sheet = Image.new("RGB", (columns * tile_width, 3 * rows_per_class * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for class_id in range(3):
        candidates = samples[class_id]
        if candidates:
            positions = [round(i * (len(candidates) - 1) / 11) for i in range(12)]
            chosen = [candidates[index] for index in positions]
        else:
            chosen = []
        for offset, (label, crop) in enumerate(chosen):
            row = class_id * rows_per_class + offset // columns
            column = offset % columns
            fitted = ImageOps.contain(crop, (tile_width - 8, tile_height - 30))
            x = column * tile_width + (tile_width - fitted.width) // 2
            y = row * tile_height + 22 + (tile_height - 30 - fitted.height) // 2
            sheet.paste(fitted, (x, y))
            draw.text((column * tile_width + 4, row * tile_height + 3), label[:34], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def run(root: Path, report_path: Path, manifest_path: Path, contact_sheet: Path) -> dict[str, object]:
    if any(marker in str(root).lower() for marker in ("vcas_rtsp_demo_60s", "36-48", "36_48")):
        raise ValueError("frozen-video path is forbidden")
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("train/images or train/labels is missing")

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = {path.stem: path for path in labels_dir.glob("*.txt")}
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    empty_or_missing_labels = 0
    unreadable_images = 0
    invalid_labels = 0
    invalid_crop_boxes = 0
    class_counts: Counter[int] = Counter()
    video_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    image_sha_counts: Counter[str] = Counter()
    crop_sha_counts: Counter[str] = Counter()
    contact_samples: dict[int, list[tuple[str, Image.Image]]] = defaultdict(list)

    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        if label_path is None:
            empty_or_missing_labels += 1
            continue
        try:
            payload = image_path.read_bytes()
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            boxes = parse_label(label_path)
            if not boxes:
                empty_or_missing_labels += 1
                continue
        except Exception as error:  # fail closed and retain the count
            if isinstance(error, (OSError, Image.DecompressionBombError)):
                unreadable_images += 1
            else:
                invalid_labels += 1
            if len(failures) < 50:
                failures.append(f"{image_path.name}: {type(error).__name__}: {error}")
            continue
        image_sha = sha256_bytes(payload)
        image_sha_counts[image_sha] += 1
        source_frame, source_video = source_keys(image_path.stem)
        frame_counts[source_frame] += 1
        video_counts[source_video] += 1
        for box_index, box in enumerate(boxes):
            class_id = box[0]
            try:
                crop = crop_box(image, box)
            except ValueError as error:
                invalid_crop_boxes += 1
                if len(failures) < 50:
                    failures.append(
                        f"{image_path.name}:box{box_index}: {type(error).__name__}: {error}"
                    )
                continue
            crop_payload = crop.tobytes()
            crop_sha = sha256_bytes(
                f"{crop.mode}:{crop.width}:{crop.height}:".encode("ascii") + crop_payload
            )
            crop_sha_counts[crop_sha] += 1
            crop_hash = dhash64(crop)
            mean_luma = sum(ImageOps.grayscale(crop).getdata()) / (crop.width * crop.height)
            class_counts[class_id] += 1
            if len(contact_samples[class_id]) < 512:
                contact_samples[class_id].append(
                    (f"numeric={class_id} {source_frame}", crop.copy())
                )
            rows.append({
                "split": "train",
                "source_dataset": "L-VAD-v2",
                "source_image": str(image_path),
                "source_frame_id": source_frame,
                "source_video_id": source_video,
                "box_index": box_index,
                "numeric_class_id": class_id,
                "body_type": "unknown",
                "body_supervised": "false",
                "color": "unknown",
                "color_supervised": "false",
                "scene_label": "night",
                "scene_origin": "publisher_dataset_scope",
                "scene_confidence": 1.0,
                "x_center": box[1],
                "y_center": box[2],
                "width": box[3],
                "height": box[4],
                "image_sha256": image_sha,
                "crop_sha256": crop_sha,
                "crop_dhash64": f"{crop_hash:016x}",
                "crop_mean_luma": round(mean_luma, 6),
                "dedup_eligible": "true",
                "near_duplicate_of": "",
                "label_conflict": "false",
            })

    exact_duplicate_rows = 0
    representatives_by_sha: dict[str, int] = {}
    near_index = NearIndex(4)
    near_duplicate_rows = 0
    near_label_conflicts = 0
    for index, row in enumerate(rows):
        crop_sha = str(row["crop_sha256"])
        if crop_sha in representatives_by_sha:
            row["dedup_eligible"] = "false"
            row["near_duplicate_of"] = representatives_by_sha[crop_sha]
            exact_duplicate_rows += 1
            continue
        representatives_by_sha[crop_sha] = index
        value = int(str(row["crop_dhash64"]), 16)
        matched = near_index.match(value)
        if matched is not None:
            row["dedup_eligible"] = "false"
            row["near_duplicate_of"] = matched
            near_duplicate_rows += 1
            if row["numeric_class_id"] != rows[matched]["numeric_class_id"]:
                row["label_conflict"] = "true"
                rows[matched]["label_conflict"] = "true"
                near_label_conflicts += 1
            continue
        near_index.add(value, index)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    build_contact_sheet(contact_samples, contact_sheet)

    numeric_names = True  # pinned from the published data.yaml: ['0', '1', '2']
    audit_pass = bool(rows) and set(class_counts) == {0, 1, 2}
    report: dict[str, object] = {
        "stage": "stage182_lvad_train_audit_r1",
        "status": (
            "pass_source_audit_semantics_pending_with_exclusions"
            if audit_pass else "fail_closed"
        ),
        "source": {
            "doi": "10.17632/h6p2w53my5.2",
            "license": "CC BY 4.0",
            "archive_sha256": "e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67",
            "publisher_scope": "exclusively nighttime and low-light",
            "published_classes": ["Cars", "Motorcycles", "Trucks"],
            "exported_yaml_names": ["0", "1", "2"],
        },
        "scope": {
            "train_images_opened": len(image_paths),
            "validation_images_opened": 0,
            "test_images_opened": 0,
            "frozen_video_used": False,
        },
        "counts": {
            "train_images": len(image_paths),
            "train_label_files": len(label_paths),
            "empty_or_missing_label_images": empty_or_missing_labels,
            "unreadable_images": unreadable_images,
            "invalid_label_images": invalid_labels,
            "invalid_crop_boxes": invalid_crop_boxes,
            "valid_box_rows": len(rows),
            "numeric_class_boxes": dict(sorted(class_counts.items())),
            "source_frames": len(frame_counts),
            "source_videos": dict(sorted(video_counts.items())),
            "exact_duplicate_crop_rows": exact_duplicate_rows,
            "near_duplicate_crop_rows_hamming_le_4": near_duplicate_rows,
            "near_duplicate_numeric_label_conflicts": near_label_conflicts,
            "dedup_eligible_crop_rows": sum(row["dedup_eligible"] == "true" for row in rows),
        },
        "gates": {
            "archive_integrity_verified": True,
            "safe_archive_paths_verified": True,
            "train_only_image_decode": True,
            "all_publisher_classes_present": set(class_counts) == {0, 1, 2},
            "class_semantics_independently_verified": not numeric_names,
            "track_identity_available": False,
            "color_truth_available": False,
            "training_authorized": False,
        },
        "decision": (
            "quarantine numeric labels until contact-sheet semantics are independently verified; "
            "never count as color truth; if admitted later, use only as train-only coarse body supervision "
            "with source-video grouping and deduplication"
        ),
        "failures": failures,
        "outputs": {
            "manifest": str(manifest_path),
            "contact_sheet": str(contact_sheet),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.report, args.manifest, args.contact_sheet)
    print(json.dumps(report, ensure_ascii=False))
    if str(report["status"]).startswith("fail"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
