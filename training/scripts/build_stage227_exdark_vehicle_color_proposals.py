#!/usr/bin/env python3
"""Detect vehicles in ExDark's official train split and make color proposals.

Only the already materialized official train images are opened.  Color values
from foreground pixels are proposals, never approved labels; a later two-model,
three-view audit must accept them or they remain unknown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
COCO_CAR = 2
COCO_BUS = 5
COCO_TRUCK = 7
LIGHTING = {
    1: "low",
    2: "ambient",
    3: "object",
    4: "single",
    5: "weak",
    6: "strong",
    7: "screen",
    8: "window",
    9: "shadow",
    10: "twilight",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def load_foreground_function(path: Path):
    spec = importlib.util.spec_from_file_location("stage227_foreground", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import foreground helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.foreground_color_evidence


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
                identity = str(
                    row.get("image_path") or row.get("crop_path")
                    or row.get("source_image") or row.get("source_frame_id") or scanned
                )
                sha = str(
                    row.get("crop_sha256") or row.get("stage177_sha256")
                    or row.get("pixel_sha256") or row.get("sha256") or ""
                ).strip().lower()
                if len(sha) == 64:
                    hashes.add(sha)
                value_text = str(
                    row.get("stage177_dhash64") or row.get("stage188_dhash64")
                    or row.get("crop_dhash64") or row.get("dhash64") or ""
                ).strip().lower()
                if len(value_text) == 16:
                    try:
                        near.add(int(value_text, 16), identity)
                    except ValueError:
                        pass
    return near, hashes, scanned


def expanded_crop(image: np.ndarray, xyxy: list[float], margin: float = 0.04) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    left, top, right, bottom = xyxy
    width, height = right - left, bottom - top
    x1 = max(0, round(left - width * margin))
    y1 = max(0, round(top - height * margin))
    x2 = min(image_width, round(right + width * margin))
    y2 = min(image_height, round(bottom + height * margin))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty crop")
    crop = image[y1:y2, x1:x2]
    maximum = max(crop.shape[:2])
    if maximum > 640:
        scale = 640 / maximum
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def choose_primary(result, expected_class: int, shape: tuple[int, int, int]):
    height, width = shape[:2]
    candidates: list[dict] = []
    if result.boxes is not None:
        for box, conf, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            cls = int(class_id)
            if cls not in {COCO_CAR, COCO_BUS, COCO_TRUCK}:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area_ratio = bw * bh / max(1.0, width * height)
            score = float(conf) * math.sqrt(max(area_ratio, 0.0))
            candidates.append(
                {
                    "xyxy": [x1, y1, x2, y2],
                    "class_id": cls,
                    "confidence": float(conf),
                    "width": bw,
                    "height": bh,
                    "area_ratio": area_ratio,
                    "score": score,
                }
            )
    matching = [item for item in candidates if item["class_id"] == expected_class]
    if not matching:
        return None, "no_source_class_agreeing_vehicle"
    matching.sort(key=lambda item: item["score"], reverse=True)
    primary = matching[0]
    if primary["confidence"] < 0.25:
        return None, "detector_confidence_below_0_25"
    if primary["area_ratio"] < 0.015 or primary["width"] < 64 or primary["height"] < 40:
        return None, "vehicle_box_too_small"
    others = sorted((item for item in candidates if item is not primary), key=lambda item: item["score"], reverse=True)
    if others and others[0]["score"] >= 0.70 * primary["score"]:
        return None, "multiple_plausible_primary_vehicles"
    return primary, "accepted"


def contact_sheet(rows: list[dict], output: Path) -> None:
    colors = ["black", "white", "silver_gray", "red", "blue", "green", "yellow_orange", "brown_beige"]
    selected: list[dict] = []
    for color in colors:
        pool = [row for row in rows if row["color"] == color and row["color_supervised"] == "true"]
        selected.extend(sorted(pool, key=lambda row: float(row["review_score"]), reverse=True)[:12])
    tile_w, tile_h, columns = 220, 170, 4
    lines = max(1, math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (tile_w * columns, tile_h * lines), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with Image.open(row["image_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 36))
        x0 = index % columns * tile_w
        y0 = index // columns * tile_h
        sheet.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 34 + (tile_h - 36 - image.height) // 2))
        draw.text((x0 + 4, y0 + 4), f"{row['color']} {row['source_image_name']}", fill="black")
        draw.text((x0 + 4, y0 + 18), f"fg={row['review_score']} det={row['detector_confidence']}", fill="black")
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--foreground-helper", type=Path, required=True)
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
    if len(source_rows) != 500 or any(row.get("official_split_id") != "1" for row in source_rows):
        raise ValueError("expected exactly 500 official train-only source rows")

    near, known_hashes, cross_rows = load_cross_source(args.cross_source_manifest, args.near_distance)
    foreground_color_evidence = load_foreground_function(args.foreground_helper)
    detector = YOLO(str(args.detector.resolve()))
    if detector.names.get(COCO_CAR) != "car" or detector.names.get(COCO_BUS) != "bus" or detector.names.get(COCO_TRUCK) != "truck":
        raise ValueError("unexpected COCO detector class contract")

    output_rows: list[dict] = []
    seen_hashes: dict[str, str] = {}
    seen_near = NearIndex(args.near_distance)
    for offset in range(0, len(source_rows), args.batch_size):
        batch = source_rows[offset: offset + args.batch_size]
        images: list[np.ndarray] = []
        decoded: list[dict] = []
        for row in batch:
            source_path = (args.input_root / row["relative_path"]).resolve()
            source_path.relative_to(args.input_root.resolve())
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                output_rows.append({
                    "split": "train", "source_image_name": row["image_name"],
                    "source_group": f"exdark:{row['image_name']}", "image_path": "",
                    "body_type": "unknown", "color": "unknown",
                    "body_type_supervised": "false", "color_supervised": "false",
                    "formal_train_eligible": "false", "stage227_exclusion_reason": "decode_failed",
                })
                continue
            images.append(image)
            decoded.append(row)
        if not images:
            continue
        results = detector.predict(
            images, imgsz=args.image_size, conf=0.20, iou=0.50,
            classes=[COCO_CAR, COCO_BUS, COCO_TRUCK], device=args.device, verbose=False,
        )
        for source, image, result in zip(decoded, images, results):
            expected = COCO_CAR if source["official_class_name"] == "car" else COCO_BUS
            primary, reason = choose_primary(result, expected, image.shape)
            base = {
                "split": "train",
                "source_dataset": "ExDark-UM-2023",
                "source_doi": "10.22452/RD/JUSQEK",
                "source_license": "CC0-repository; research-only-effective-policy",
                "license_train_eligible": "true",
                "research_only": "true",
                "deployment_eligible": "false",
                "source_image_name": source["image_name"],
                "source_image_id": source["image_name"],
                "source_group": f"exdark:{source['image_name']}",
                "source_class": source["official_class_name"],
                "official_lighting_id": source["lighting_id"],
                "lighting": LIGHTING[int(source["lighting_id"])],
                "scene_label": "source_low_light_visible_spectrum",
                "confirmed_real_night": "false",
                "image_path": "",
                "body_type": "unknown",
                "color": "unknown",
                "body_type_supervised": "false",
                "color_supervised": "false",
                "formal_train_eligible": "false",
                "pseudo_label": "false",
                "annotation_source": "none",
                "review_status": "rejected" if reason != "accepted" else "pending_foreground",
                "color_review_status": "unknown",
                "review_method": "",
                "review_score": "0.000000",
                "detector_class_id": "",
                "detector_confidence": "",
                "detector_area_ratio": "",
                "crop_width": "",
                "crop_height": "",
                "crop_mean_luma": "",
                "crop_contrast": "",
                "crop_sharpness": "",
                "crop_sha256": "",
                "crop_dhash64": "",
                "cross_source_duplicate_of": "",
                "internal_duplicate_of": "",
                "foreground_margin": "",
                "foreground_mean_value": "",
                "foreground_mean_saturation": "",
                "stage227_exclusion_reason": reason,
            }
            if primary is None:
                output_rows.append(base)
                continue
            crop = expanded_crop(image, primary["xyxy"])
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            contrast = float(gray.std())
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            if mean < 4 or contrast < 5 or sharpness < 3:
                base["stage227_exclusion_reason"] = "insufficient_visible_color_signal"
                base["review_status"] = "rejected"
                output_rows.append(base)
                continue
            encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encoded_ok:
                base["stage227_exclusion_reason"] = "jpeg_encode_failed"
                output_rows.append(base)
                continue
            crop_name = hashlib.sha1(source["image_name"].encode("utf-8")).hexdigest()[:20] + ".jpg"
            crop_path = crop_root / crop_name
            crop_path.write_bytes(encoded.tobytes())
            sha = digest(crop_path)
            dhash = dhash64(crop)
            base.update({
                "image_path": str(crop_path.resolve()),
                "detector_class_id": str(primary["class_id"]),
                "detector_confidence": f"{primary['confidence']:.6f}",
                "detector_area_ratio": f"{primary['area_ratio']:.6f}",
                "crop_width": str(crop.shape[1]),
                "crop_height": str(crop.shape[0]),
                "crop_mean_luma": f"{mean:.6f}",
                "crop_contrast": f"{contrast:.6f}",
                "crop_sharpness": f"{sharpness:.6f}",
                "crop_sha256": sha,
                "crop_dhash64": f"{dhash:016x}",
            })
            if sha in known_hashes:
                base["cross_source_duplicate_of"] = "exact_sha256"
                base["stage227_exclusion_reason"] = "cross_source_exact_duplicate"
                base["review_status"] = "rejected"
                output_rows.append(base)
                continue
            matched = near.match(dhash)
            if matched is not None:
                base["cross_source_duplicate_of"] = matched
                base["stage227_exclusion_reason"] = "cross_source_perceptual_duplicate"
                base["review_status"] = "rejected"
                output_rows.append(base)
                continue
            if sha in seen_hashes:
                base["internal_duplicate_of"] = seen_hashes[sha]
                base["stage227_exclusion_reason"] = "internal_exact_duplicate"
                base["review_status"] = "rejected"
                output_rows.append(base)
                continue
            internal_near = seen_near.match(dhash)
            if internal_near is not None:
                base["internal_duplicate_of"] = internal_near
                base["stage227_exclusion_reason"] = "internal_perceptual_duplicate"
                base["review_status"] = "rejected"
                output_rows.append(base)
                continue
            seen_hashes[sha] = source["image_name"]
            seen_near.add(dhash, source["image_name"])
            evidence = foreground_color_evidence(crop_path)
            proposal = str(evidence.get("label", "unknown"))
            base["foreground_margin"] = f"{float(evidence.get('margin', 0.0)):.6f}"
            base["foreground_mean_value"] = f"{float(evidence.get('mean_value', 0.0)):.6f}"
            base["foreground_mean_saturation"] = f"{float(evidence.get('mean_saturation', 0.0)):.6f}"
            if proposal == "unknown":
                base["stage227_exclusion_reason"] = f"foreground_{evidence.get('reason', 'unknown')}"
                base["review_status"] = "rejected"
            else:
                base.update({
                    "color": proposal,
                    "color_supervised": "true",
                    "pseudo_label": "true",
                    "annotation_source": "foreground_pixels_proposal_only",
                    "review_status": "pending_teacher_audit",
                    "color_review_status": "foreground_proposal_pending_teacher_audit",
                    "review_method": "foreground_pixels_only_proposal_not_approved_label",
                    "review_score": f"{float(evidence.get('score', 0.0)):.6f}",
                    "stage227_exclusion_reason": "",
                })
            output_rows.append(base)

    if len(output_rows) != 500:
        raise RuntimeError(f"expected 500 terminal rows, got {len(output_rows)}")
    manifest = args.output_root / "stage227-exdark-color-proposals.csv"
    fields = list(output_rows[0])
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    sheet = args.output_root / "stage227-exdark-color-proposals-contact-sheet.jpg"
    contact_sheet(output_rows, sheet)
    exclusions = Counter(row["stage227_exclusion_reason"] or "proposal" for row in output_rows)
    proposals = Counter(row["color"] for row in output_rows if row["color_supervised"] == "true")
    report = {
        "stage": "stage227-exdark-vehicle-color-proposals-r1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_pending_multiteacher_and_visual_audit",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": digest(args.input_manifest),
        "source_train_images": len(source_rows),
        "terminal_rows": len(output_rows),
        "materialized_crops": sum(bool(row["image_path"]) for row in output_rows),
        "foreground_proposals": sum(proposals.values()),
        "foreground_proposal_color_counts": dict(sorted(proposals.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "cross_source_rows_scanned": cross_rows,
        "cross_source_exact_duplicates": sum(row["stage227_exclusion_reason"] == "cross_source_exact_duplicate" for row in output_rows),
        "cross_source_perceptual_duplicates": sum(row["stage227_exclusion_reason"] == "cross_source_perceptual_duplicate" for row in output_rows),
        "internal_duplicates": sum(bool(row["internal_duplicate_of"]) for row in output_rows),
        "detector": str(args.detector.resolve()),
        "detector_sha256": digest(args.detector),
        "foreground_helper": str(args.foreground_helper.resolve()),
        "foreground_helper_sha256": digest(args.foreground_helper),
        "policy": {
            "official_train_images_only": True,
            "validation_pixels_opened": 0,
            "test_pixels_opened": 0,
            "frozen_video_used": False,
            "proposals_are_approved_labels": False,
            "multiteacher_and_visual_audit_required": True,
            "confirmed_real_night_rows": 0,
            "research_only": True,
            "deployment_allowed": False,
            "production_model_modified": False,
            "training_started": False,
        },
        "outputs": {
            "manifest": str(manifest),
            "contact_sheet": str(sheet),
            "crop_root": str(crop_root),
        },
    }
    report_path = args.output_root / "stage227-exdark-vehicle-color-proposals.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in (manifest, sheet, report_path):
            handle.write(f"{digest(path)}  {path}\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
