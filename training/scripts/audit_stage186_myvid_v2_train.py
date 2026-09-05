#!/usr/bin/env python3
"""Re-audit extracted MY-VID v2 train images and keep labels quarantined."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_RE = re.compile(r"^(?P<frame>.+?)_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
EXPORT_AUGMENTATION_RE = re.compile(r"_(?P<kind>flip_h|flip_v|gray)$", re.IGNORECASE)
VIDEO_RE = re.compile(r"^(?P<video>.*?)(?:_?\d+)$")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")
PUBLISHER_CLASSES = [
    "cars_and_taxis",
    "vans_and_utilities",
    "medium_lorries",
    "heavy_lorries",
    "buses",
    "motorcycles",
]


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


def source_keys(stem: str) -> tuple[str, str, str]:
    match = SOURCE_RE.match(stem)
    frame = match.group("frame") if match else stem
    augmentation_match = EXPORT_AUGMENTATION_RE.search(frame)
    augmentation = augmentation_match.group("kind").lower() if augmentation_match else "original"
    source_frame = frame[:augmentation_match.start()] if augmentation_match else frame
    video_match = VIDEO_RE.match(source_frame)
    video = video_match.group("video") if video_match else source_frame
    return source_frame, video.rstrip("_") or "unknown", augmentation


def yaml_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^names:\s*(\[[^\n]+\])\s*$", text)
    if not match:
        return []
    value = ast.literal_eval(match.group(1))
    return [str(item) for item in value] if isinstance(value, list) else []


def parse_label(path: Path, class_count: int) -> list[tuple[int, float, float, float, float, str]]:
    """Parse YOLO boxes or segmentation polygons and return normalized boxes.

    MY-VID v2 is advertised as YOLO text, but the released train labels use
    segmentation polygons (class followed by x/y pairs).  Converting those
    polygons to an axis-aligned box is lossless for this crop-only audit and
    leaves the source label untouched.
    """
    boxes: list[tuple[int, float, float, float, float, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5 and (len(fields) < 7 or len(fields) % 2 == 0):
            raise ValueError(
                f"{path.name}:{line_number}: expected YOLO box or polygon, got {len(fields)} fields"
            )
        class_value = float(fields[0])
        class_id = int(class_value)
        values = tuple(float(value) for value in fields[1:])
        if class_value != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path.name}:{line_number}: invalid class {fields[0]}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{line_number}: non-finite box")
        if len(fields) == 5:
            x, y, width, height = values
            annotation_format = "yolo_box"
        else:
            xs, ys = values[0::2], values[1::2]
            if len(xs) < 3:
                raise ValueError(f"{path.name}:{line_number}: polygon has fewer than 3 points")
            if not all(-1e-4 <= value <= 1.0001 for value in values):
                raise ValueError(f"{path.name}:{line_number}: polygon point outside image")
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            width, height = right - left, bottom - top
            x, y = (left + right) / 2, (top + bottom) / 2
            annotation_format = "yolo_segmentation_polygon"
        if width <= 0 or height <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{path.name}:{line_number}: invalid box")
        if x - width / 2 < -1e-4 or x + width / 2 > 1.0001:
            raise ValueError(f"{path.name}:{line_number}: horizontal box outside image")
        if y - height / 2 < -1e-4 or y + height / 2 > 1.0001:
            raise ValueError(f"{path.name}:{line_number}: vertical box outside image")
        boxes.append((class_id, x, y, width, height, annotation_format))
    return boxes


def crop_box(image: Image.Image, box: tuple[int, float, float, float, float, str]) -> Image.Image:
    _, x, y, width, height, _ = box
    image_width, image_height = image.size
    left = max(0, round((x - width / 2) * image_width))
    top = max(0, round((y - height / 2) * image_height))
    right = min(image_width, round((x + width / 2) * image_width))
    bottom = min(image_height, round((y + height / 2) * image_height))
    if right <= left or bottom <= top:
        raise ValueError("empty pixel crop")
    return image.crop((left, top, right, bottom))


def frame_metrics(image: Image.Image) -> dict[str, float | bool]:
    gray = np.asarray(ImageOps.grayscale(image).resize((192, 108), Image.Resampling.BILINEAR), dtype=np.float32)
    border_y, border_x = max(2, round(gray.shape[0] * 0.12)), max(2, round(gray.shape[1] * 0.12))
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
    score = (
        0.28 * float(np.clip((95.0 - mean) / 65.0, 0.0, 1.0))
        + 0.22 * float(np.clip((90.0 - median) / 65.0, 0.0, 1.0))
        + 0.25 * float(np.clip((100.0 - border_mean) / 70.0, 0.0, 1.0))
        + 0.15 * min(1.0, dark_fraction / 0.60)
        + 0.10 * float(np.clip((165.0 - p90) / 105.0, 0.0, 1.0))
    )
    lowlight = score >= 0.62 and mean <= 88 and median <= 82 and border_mean <= 95 and dark_fraction >= 0.30
    night_candidate = score >= 0.78 and mean <= 70 and median <= 65 and border_mean <= 80 and dark_fraction >= 0.45
    return {
        "mean_luma": mean,
        "median_luma": median,
        "border_mean_luma": border_mean,
        "dark_fraction": dark_fraction,
        "lowlight_score": score,
        "frame_lowlight_candidate": lowlight,
        "frame_night_candidate": night_candidate,
    }


def contact_sheet(samples: dict[int, list[tuple[str, Image.Image]]], output: Path, class_count: int) -> None:
    tile_width, tile_height, columns, rows_per_class = 220, 170, 4, 3
    sheet = Image.new("RGB", (columns * tile_width, class_count * rows_per_class * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for class_id in range(class_count):
        candidates = samples[class_id]
        positions = [round(i * (len(candidates) - 1) / 11) for i in range(12)] if candidates else []
        for offset, index in enumerate(positions):
            label, crop = candidates[index]
            row = class_id * rows_per_class + offset // columns
            column = offset % columns
            fitted = ImageOps.contain(crop, (tile_width - 8, tile_height - 30))
            x = column * tile_width + (tile_width - fitted.width) // 2
            y = row * tile_height + 22 + (tile_height - 30 - fitted.height) // 2
            sheet.paste(fitted, (x, y))
            draw.text((column * tile_width + 4, row * tile_height + 3), label[:34], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def run(root: Path, report_path: Path, manifest_path: Path, contact_path: Path) -> dict[str, object]:
    if any(marker in str(root).lower() for marker in FROZEN_MARKERS):
        raise ValueError("frozen-video path is forbidden")
    names = yaml_names(root / "data.yaml")
    class_count = len(names)
    images_dir, labels_dir = root / "train" / "images", root / "train" / "labels"
    if class_count != 6 or not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"expected six names and train directories; names={names}")

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = {path.stem: path for path in labels_dir.glob("*.txt")}
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    class_counts: Counter[int] = Counter()
    video_frame_counts: Counter[str] = Counter()
    video_lowlight_counts: Counter[str] = Counter()
    empty_or_missing = unreadable = invalid_labels = invalid_crops = 0
    frame_lowlight = frame_night = 0
    original_frame_lowlight = original_frame_night = 0
    augmentation_counts: Counter[str] = Counter()
    annotation_format_counts: Counter[str] = Counter()
    unique_source_frames: set[str] = set()
    samples: dict[int, list[tuple[str, Image.Image]]] = defaultdict(list)

    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        if label_path is None:
            empty_or_missing += 1
            continue
        try:
            payload = image_path.read_bytes()
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            boxes = parse_label(label_path, class_count)
            if not boxes:
                empty_or_missing += 1
                continue
            metrics = frame_metrics(image)
        except OSError as error:
            unreadable += 1
            if len(failures) < 50:
                failures.append(f"{image_path.name}: {type(error).__name__}: {error}")
            continue
        except ValueError as error:
            invalid_labels += 1
            if len(failures) < 50:
                failures.append(f"{image_path.name}: {type(error).__name__}: {error}")
            continue
        source_frame, source_video, export_augmentation = source_keys(image_path.stem)
        augmentation_counts[export_augmentation] += 1
        unique_source_frames.add(source_frame)
        if export_augmentation == "original":
            video_frame_counts[source_video] += 1
        if metrics["frame_lowlight_candidate"]:
            frame_lowlight += 1
            if export_augmentation == "original":
                original_frame_lowlight += 1
                video_lowlight_counts[source_video] += 1
        if metrics["frame_night_candidate"]:
            frame_night += 1
            if export_augmentation == "original":
                original_frame_night += 1
        image_sha = hashlib.sha256(payload).hexdigest()
        for box_index, box in enumerate(boxes):
            try:
                crop = crop_box(image, box)
            except ValueError as error:
                invalid_crops += 1
                if len(failures) < 50:
                    failures.append(f"{image_path.name}:box{box_index}: {error}")
                continue
            class_id = box[0]
            annotation_format = box[5]
            annotation_format_counts[annotation_format] += 1
            crop_payload = crop.tobytes()
            crop_sha = hashlib.sha256(
                f"{crop.mode}:{crop.width}:{crop.height}:".encode("ascii") + crop_payload
            ).hexdigest()
            crop_hash = dhash64(crop)
            crop_luma = float(np.asarray(ImageOps.grayscale(crop), dtype=np.float32).mean())
            class_counts[class_id] += 1
            if len(samples[class_id]) < 1024:
                samples[class_id].append((f"numeric={class_id} {source_frame}", crop.copy()))
            rows.append({
                "split": "train",
                "source_dataset": "MY-VID-v2",
                "source_image": str(image_path),
                "source_frame_id": source_frame,
                "source_video_id": source_video,
                "export_augmentation": export_augmentation,
                "box_index": box_index,
                "numeric_class_id": class_id,
                "yaml_class_name": names[class_id],
                "publisher_class_name": PUBLISHER_CLASSES[class_id],
                "annotation_format": annotation_format,
                "body_type": "unknown",
                "coarse_body_family": "unknown",
                "body_supervised": "false",
                "color": "unknown",
                "color_supervised": "false",
                "scene_label": "unknown",
                "scene_origin": "publisher_mixed_day_night_no_per_frame_truth",
                "scene_confidence": 0.0,
                "frame_lowlight_candidate": str(metrics["frame_lowlight_candidate"]).lower(),
                "frame_night_candidate": str(metrics["frame_night_candidate"]).lower(),
                "frame_lowlight_score": round(float(metrics["lowlight_score"]), 6),
                "x_center": box[1], "y_center": box[2], "width": box[3], "height": box[4],
                "image_sha256": image_sha,
                "crop_sha256": crop_sha,
                "crop_dhash64": f"{crop_hash:016x}",
                "crop_mean_luma": round(crop_luma, 6),
                "dedup_eligible": "true",
                "near_duplicate_of": "",
                "label_conflict": "false",
            })

    qualifying_lowlight_videos = {
        video for video, total in video_frame_counts.items()
        if video_lowlight_counts[video] >= 3 and video_lowlight_counts[video] / total >= 0.25
    }
    for row in rows:
        if row["source_video_id"] in qualifying_lowlight_videos and row["frame_lowlight_candidate"] == "true":
            row["scene_label"] = "low_light"
            row["scene_origin"] = "machine_video_consistent_lowlight_proxy"
            row["scene_confidence"] = 0.75

    exact_duplicates = near_duplicates = near_conflicts = 0
    representatives: dict[str, int] = {}
    near_index = NearIndex(4)
    for index, row in enumerate(rows):
        sha = str(row["crop_sha256"])
        if sha in representatives:
            row["dedup_eligible"] = "false"
            row["near_duplicate_of"] = representatives[sha]
            exact_duplicates += 1
            continue
        representatives[sha] = index
        value = int(str(row["crop_dhash64"]), 16)
        matched = near_index.match(value)
        if matched is not None:
            row["dedup_eligible"] = "false"
            row["near_duplicate_of"] = matched
            near_duplicates += 1
            if row["numeric_class_id"] != rows[matched]["numeric_class_id"]:
                row["label_conflict"] = rows[matched]["label_conflict"] = "true"
                near_conflicts += 1
            continue
        near_index.add(value, index)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    contact_sheet(samples, contact_path, class_count)

    target_class_ids = set(range(5))
    audit_pass = bool(rows) and target_class_ids.issubset(class_counts) and set(class_counts).issubset(set(range(6)))
    promoted_frames = sum(
        1 for image_path in image_paths
        if source_keys(image_path.stem)[2] == "original"
        and source_keys(image_path.stem)[1] in qualifying_lowlight_videos
    )
    report: dict[str, object] = {
        "stage": "stage187_myvid_v2_train_reaudit_r1",
        "status": "pass_semantics_and_scene_truth_pending_with_exclusions" if audit_pass else "fail_closed",
        "source": {
            "doi": "10.5281/zenodo.17861464",
            "license": "CC BY 4.0",
            "yaml_names": names,
            "publisher_class_names": PUBLISHER_CLASSES,
            "publisher_class_source": "Zenodo record 17861464 description",
            "publisher_scope": "mixed day and night; no per-frame time-of-day truth admitted by this audit",
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
            "empty_or_missing_label_images": empty_or_missing,
            "unreadable_images": unreadable,
            "invalid_label_images": invalid_labels,
            "invalid_crop_boxes": invalid_crops,
            "valid_box_rows": len(rows),
            "numeric_class_boxes": dict(sorted(class_counts.items())),
            "annotation_formats": dict(sorted(annotation_format_counts.items())),
            "export_augmentation_images": dict(sorted(augmentation_counts.items())),
            "unique_source_frames_before_cross_split_audit": len(unique_source_frames),
            "source_videos": dict(sorted(video_frame_counts.items())),
            "machine_lowlight_candidate_frames": frame_lowlight,
            "machine_night_candidate_frames_excluded_from_confirmed_quota": frame_night,
            "machine_lowlight_candidate_original_frames": original_frame_lowlight,
            "machine_night_candidate_original_frames_excluded_from_confirmed_quota": original_frame_night,
            "machine_video_consistent_lowlight_groups": sorted(qualifying_lowlight_videos),
            "machine_video_consistent_lowlight_group_frames": promoted_frames,
            "confirmed_night_frames": 0,
            "exact_duplicate_crop_rows": exact_duplicates,
            "near_duplicate_crop_rows_hamming_le_4": near_duplicates,
            "near_duplicate_numeric_label_conflicts": near_conflicts,
            "dedup_eligible_crop_rows": sum(row["dedup_eligible"] == "true" for row in rows),
        },
        "gates": {
            "six_class_taxonomy_declared": len(PUBLISHER_CLASSES) == 6,
            "all_five_in_scope_vehicle_classes_present_in_train": target_class_ids.issubset(class_counts),
            "motorcycle_class_present_in_train": 5 in class_counts,
            "class_semantics_independently_verified": False,
            "per_frame_night_truth_available": False,
            "track_identity_available": False,
            "color_truth_available": False,
            "training_authorized": False,
        },
        "decision": "retain all labels quarantined until visual class mapping, temporal/source grouping, cross-source decontamination and scene audit pass; add no color truth and no confirmed-night truth",
        "failures": failures,
        "outputs": {"manifest": str(manifest_path), "contact_sheet": str(contact_path)},
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
    if report["status"] == "fail_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
