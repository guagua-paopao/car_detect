#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class NearIndex:
    def __init__(self, distance: int) -> None:
        self.distance = distance
        self.bands: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)

    def add(self, value: int, identity: str) -> None:
        for band_index in range(5):
            width = 13 if band_index < 4 else 12
            offset = band_index * 13
            mask = (1 << width) - 1
            self.bands[(band_index, (value >> offset) & mask)].append((value, identity))

    def match(self, value: int) -> str | None:
        candidates: set[tuple[int, str]] = set()
        for band_index in range(5):
            width = 13 if band_index < 4 else 12
            offset = band_index * 13
            mask = (1 << width) - 1
            candidates.update(self.bands.get((band_index, (value >> offset) & mask), []))
        for other, identity in candidates:
            if (value ^ other).bit_count() <= self.distance:
                return identity
        return None


def dhash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def load_cross_source(paths: list[Path], distance: int) -> tuple[NearIndex, set[str], int]:
    near = NearIndex(distance)
    hashes: set[str] = set()
    scanned = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("split") or "train").strip().lower() != "train":
                    continue
                scanned += 1
                identity = str(row.get("image_path") or row.get("crop_path") or row.get("source_frame_id") or scanned)
                sha = str(row.get("crop_sha256") or row.get("stage177_sha256") or row.get("pixel_sha256") or row.get("sha256") or "").lower()
                if len(sha) == 64:
                    hashes.add(sha)
                text = str(row.get("stage177_dhash64") or row.get("stage188_dhash64") or row.get("crop_dhash64") or row.get("dhash64") or "").lower()
                if len(text) == 16:
                    try:
                        near.add(int(text, 16), identity)
                    except ValueError:
                        pass
    return near, hashes, scanned


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def assign_tracks(frames: list[list[dict]], threshold: float = 0.15, max_gap: int = 2) -> int:
    active: dict[int, dict] = {}
    next_id = 0
    for frame_index, detections in enumerate(frames):
        pairs = []
        for track_id, last in active.items():
            if frame_index - last["frame_index"] > max_gap:
                continue
            for detection_index, detection in enumerate(detections):
                if detection["class_id"] != last["class_id"]:
                    continue
                overlap = iou(last["xyxy"], detection["xyxy"])
                if overlap >= threshold:
                    pairs.append((overlap, track_id, detection_index))
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_id, detection_index in sorted(pairs, reverse=True):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            detections[detection_index]["track_id"] = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned_detections:
                detection["track_id"] = next_id
                next_id += 1
            active[detection["track_id"]] = {
                "xyxy": detection["xyxy"],
                "class_id": detection["class_id"],
                "frame_index": frame_index,
            }
    return next_id


def crop_with_margin(image: np.ndarray, box: list[float], margin: float = 0.04) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    touches = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
    left = max(0, round(x1 - bw * margin))
    top = max(0, round(y1 - bh * margin))
    right = min(width, round(x2 + bw * margin))
    bottom = min(height, round(y2 + bh * margin))
    if right <= left or bottom <= top:
        raise ValueError("empty crop")
    crop = image[top:bottom, left:right]
    maximum = max(crop.shape[:2])
    if maximum > 512:
        scale = 512.0 / maximum
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return crop, touches


def make_contact_sheet(rows: list[dict], output: Path) -> None:
    eligible = [row for row in rows if row["crop_quality_usable"] == "true"]
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_track[row["track_key"]].append(row)
    selected = []
    for track in sorted(by_track):
        selected.append(max(by_track[track], key=lambda item: float(item["quality_score"])))
    selected = sorted(selected, key=lambda item: float(item["quality_score"]), reverse=True)[:80]
    tile_w, tile_h, columns = 240, 180, 5
    sheet = Image.new("RGB", (tile_w * columns, tile_h * max(1, math.ceil(len(selected) / columns))), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with Image.open(row["crop_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 42))
        x0 = index % columns * tile_w
        y0 = index // columns * tile_h
        sheet.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 38 + (tile_h - 42 - image.height) // 2))
        draw.text((x0 + 4, y0 + 4), f"{row['detector_class']} conf={row['detector_confidence']}", fill="black")
        draw.text((x0 + 4, y0 + 19), f"{row['track_key'].split('|')[-1]} len={row['track_length']}", fill="black")
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--cross-source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--near-distance", type=int, default=2)
    args = parser.parse_args()
    for value in vars(args).values():
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video marker is forbidden")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    crop_root = args.output_root / "crops"
    crop_root.mkdir()

    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 2400:
        raise ValueError(f"expected 2400 train rows, got {len(source_rows)}")
    if any(row.get("daytime") != "night" for row in source_rows):
        raise ValueError("non-night row found")
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[(int(row["recording_id"]), int(row["block_index"]))].append(row)
    if len(grouped) != 120 or any(len(rows) != 20 for rows in grouped.values()):
        raise ValueError("expected 120 groups of 20 contiguous train frames")

    cross_near, cross_hashes, cross_rows = load_cross_source(args.cross_source_manifest, args.near_distance)
    internal_near = NearIndex(args.near_distance)
    internal_hashes: dict[str, str] = {}
    detector = YOLO(str(args.detector.resolve()))
    for class_id, name in VEHICLE_CLASSES.items():
        if detector.names.get(class_id) != name:
            raise ValueError("unexpected COCO detector class contract")

    output_rows: list[dict] = []
    frame_errors = []
    frames_without_vehicle = 0
    for group_ordinal, ((recording, block), rows) in enumerate(sorted(grouped.items()), start=1):
        rows = sorted(rows, key=lambda item: (int(item["timestamp"]), int(item["image_id"])))
        images = []
        for row in rows:
            path = Path(row["local_path"])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (640, 1024):
                frame_errors.append({"path": str(path), "reason": "decode_or_shape_failure"})
                images.append(None)
            else:
                images.append(image)
        detections_by_frame: list[list[dict]] = [[] for _ in rows]
        valid_indices = [index for index, image in enumerate(images) if image is not None]
        for offset in range(0, len(valid_indices), args.batch_size):
            indices = valid_indices[offset : offset + args.batch_size]
            results = detector.predict(
                [images[index] for index in indices], imgsz=args.image_size, conf=0.20, iou=0.50,
                classes=list(VEHICLE_CLASSES), device=args.device, verbose=False,
            )
            for frame_index, result in zip(indices, results):
                height, width = images[frame_index].shape[:2]
                if result.boxes is None:
                    continue
                for box, confidence, class_value in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    class_id = int(class_value)
                    x1, y1, x2, y2 = [float(value) for value in box]
                    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    area_ratio = bw * bh / max(1.0, width * height)
                    if float(confidence) < 0.25 or bw < 28 or bh < 20 or area_ratio < 0.0008:
                        continue
                    detections_by_frame[frame_index].append({
                        "class_id": class_id,
                        "confidence": float(confidence),
                        "xyxy": [x1, y1, x2, y2],
                        "area_ratio": area_ratio,
                    })
        frames_without_vehicle += sum(not items for items in detections_by_frame)
        assign_tracks(detections_by_frame)
        group_prefix = f"NightOwls|train|recording_{recording:02d}|block_{block:02d}"
        for frame_index, (source, image, detections) in enumerate(zip(rows, images, detections_by_frame)):
            if image is None:
                continue
            for detection_index, detection in enumerate(detections):
                track_key = f"{group_prefix}|track_{detection['track_id']:03d}"
                crop, border_truncated = crop_with_margin(image, detection["xyxy"])
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean_luma = float(gray.mean())
                contrast = float(gray.std())
                sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mean_saturation = float(hsv[:, :, 1].mean())
                quality_usable = mean_luma >= 8 and contrast >= 8 and sharpness >= 4 and min(crop.shape[:2]) >= 20
                quality_score = (
                    detection["confidence"] * math.sqrt(detection["area_ratio"])
                    * min(1.0, contrast / 40.0) * min(1.0, sharpness / 80.0)
                    * (0.75 if border_truncated else 1.0)
                )
                crop_name = f"f{frame_index:02d}_d{detection_index:02d}_{source['image_id']}.jpg"
                target_dir = crop_root / f"recording_{recording:02d}" / f"block_{block:02d}" / f"track_{detection['track_id']:03d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                crop_path = target_dir / crop_name
                if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(f"failed to write crop: {crop_path}")
                sha = digest(crop_path)
                dhash = dhash64(crop)
                cross_duplicate = ""
                internal_duplicate = ""
                if sha in cross_hashes:
                    cross_duplicate = "exact_sha256"
                else:
                    cross_duplicate = cross_near.match(dhash) or ""
                if sha in internal_hashes:
                    internal_duplicate = internal_hashes[sha]
                else:
                    internal_duplicate = internal_near.match(dhash) or ""
                identity = str(crop_path)
                internal_hashes.setdefault(sha, identity)
                internal_near.add(dhash, identity)
                size_bin = "small" if detection["area_ratio"] < 0.01 else ("medium" if detection["area_ratio"] < 0.05 else "large")
                output_rows.append({
                    "split": "train",
                    "source_dataset": "NightOwls",
                    "source_license": "non-commercial research-only",
                    "research_only": "true",
                    "deployment_eligible": "false",
                    "recording_id": recording,
                    "block_index": block,
                    "frame_index": frame_index,
                    "source_image_id": source["image_id"],
                    "source_image_path": source["local_path"],
                    "source_frame_sha256": source["sha256"],
                    "group_key": group_prefix,
                    "track_key": track_key,
                    "detector_class_id": detection["class_id"],
                    "detector_class": VEHICLE_CLASSES[detection["class_id"]],
                    "detector_confidence": f"{detection['confidence']:.6f}",
                    "detector_area_ratio": f"{detection['area_ratio']:.8f}",
                    "size_bin": size_bin,
                    "border_truncated": str(border_truncated).lower(),
                    "crop_path": str(crop_path.resolve()),
                    "crop_width": crop.shape[1],
                    "crop_height": crop.shape[0],
                    "crop_mean_luma": f"{mean_luma:.6f}",
                    "crop_contrast": f"{contrast:.6f}",
                    "crop_sharpness": f"{sharpness:.6f}",
                    "crop_mean_saturation": f"{mean_saturation:.6f}",
                    "quality_score": f"{quality_score:.8f}",
                    "crop_quality_usable": str(quality_usable).lower(),
                    "crop_sha256": sha,
                    "crop_dhash64": f"{dhash:016x}",
                    "cross_source_duplicate_of": cross_duplicate,
                    "internal_near_duplicate_of": internal_duplicate,
                    "scene_label": "source_confirmed_night_visible_spectrum",
                    "body_type": "unknown",
                    "color": "unknown",
                    "body_type_supervised": "false",
                    "color_supervised": "false",
                    "formal_train_eligible": "false",
                })
        print(json.dumps({"groups_done": group_ordinal, "groups_total": len(grouped), "crops": len(output_rows)}), flush=True)

    track_lengths = Counter(row["track_key"] for row in output_rows)
    for row in output_rows:
        row["track_length"] = track_lengths[row["track_key"]]
    manifest = args.output_root / "stage234-nightowls-vehicle-tracks.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    sheet = args.output_root / "stage234-nightowls-vehicle-tracks-contact-sheet.jpg"
    make_contact_sheet(output_rows, sheet)
    report = {
        "schema_version": "stage234-nightowls-vehicle-tracks-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_pending_color_multiteacher_audit",
        "source_train_frames": len(source_rows),
        "recording_groups": len({row["recording_id"] for row in source_rows}),
        "block_groups": len(grouped),
        "frame_errors": len(frame_errors),
        "frames_without_vehicle": frames_without_vehicle,
        "detected_vehicle_crops": len(output_rows),
        "vehicle_tracks": len(track_lengths),
        "tracks_with_at_least_3_frames": sum(length >= 3 for length in track_lengths.values()),
        "tracks_with_at_least_5_frames": sum(length >= 5 for length in track_lengths.values()),
        "detector_class_counts": dict(sorted(Counter(row["detector_class"] for row in output_rows).items())),
        "size_bin_counts": dict(sorted(Counter(row["size_bin"] for row in output_rows).items())),
        "border_truncated_crops": sum(row["border_truncated"] == "true" for row in output_rows),
        "quality_usable_crops": sum(row["crop_quality_usable"] == "true" for row in output_rows),
        "cross_source_duplicates": sum(bool(row["cross_source_duplicate_of"]) for row in output_rows),
        "internal_near_duplicates": sum(bool(row["internal_near_duplicate_of"]) for row in output_rows),
        "cross_source_rows_scanned": cross_rows,
        "detector": str(args.detector.resolve()),
        "detector_sha256": digest(args.detector),
        "inputs": {
            "manifest": str(args.input_manifest.resolve()),
            "manifest_sha256": digest(args.input_manifest),
        },
        "outputs": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": digest(manifest),
            "contact_sheet": str(sheet.resolve()),
            "contact_sheet_sha256": digest(sheet),
            "crop_root": str(crop_root.resolve()),
        },
        "policy": {
            "official_train_frames_only": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "labels_generated": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage234-nightowls-vehicle-tracks-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in (manifest, sheet, report_path):
            handle.write(f"{digest(path)}  {path}\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
