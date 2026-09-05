#!/usr/bin/env python3
"""Audit a CC0 Kaggle vehicle-color archive and build train-only color crops.

The archive is treated fail-closed.  Only explicit, unobstructed four-wheel
vehicle boxes with known single-color labels are accepted.  Motorcycles,
scooters, auto-rickshaws, ambiguous labels, unreadable boxes and conflicting
near-duplicate groups are excluded.  No model predictions are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageFilter, ImageStat


COLOR_MAP = {
    "black": "black", "white": "white", "grey": "silver_gray", "gray": "silver_gray",
    "silver": "silver_gray", "blue": "blue", "red": "red", "green": "green",
    "yellow": "yellow_orange", "orange": "yellow_orange", "brown": "brown_beige",
    "beige": "brown_beige",
}
ACCEPTED_TYPES = {"car", "truck", "van", "tempo", "bus"}
EXCLUDED_TYPES = {"bike", "scooty", "scooter", "motorcycle", "auto", "rickshaw"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_xml(node: ET.Element, key: str) -> bool:
    value = node.findtext(key, "0").strip().lower()
    return value not in {"", "0", "false", "no"}


def parse_label(value: str) -> tuple[str, str]:
    text = value.strip().lower()
    parts = text.split("_", 1)
    if len(parts) != 2:
        return "unknown", "unknown"
    return COLOR_MAP.get(parts[0], "unknown"), parts[1]


def contributor_group(image_name: str) -> str:
    stem = PurePosixPath(image_name).stem
    match = re.match(r"^[0-9]{8}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3}_([^_]+)_[A-Za-z]_", stem)
    return match.group(1) if match else stem


def capture_sequence_group(image_name: str) -> str:
    """Return a narrow capture-burst group for perceptual deduplication."""
    stem = PurePosixPath(image_name).stem
    parts = stem.split("_")
    contributor = contributor_group(image_name)
    if len(parts) >= 4 and re.fullmatch(r"[0-9]{8}", parts[0]):
        return f"{contributor}:{parts[0]}:{parts[1]}:{parts[2]}"
    return stem


def quality(image: Image.Image) -> tuple[float, float, float]:
    gray = image.convert("L").resize((128, 128))
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    contrast = math.sqrt(float(stats.var[0]))
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = float(ImageStat.Stat(edge).var[0])
    return mean, contrast, edge_variance


def normalized_sha(image: Image.Image) -> str:
    normalized = image.convert("RGB").resize((128, 128))
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8))
    pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def paired_images(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result[str(PurePosixPath(name).with_suffix(""))] = name
    return result


def load_crop(archive: zipfile.ZipFile, candidate: dict) -> Image.Image:
    with archive.open(candidate["image_zip_path"]) as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
    expected_size = (candidate["declared_width"], candidate["declared_height"])
    if image.size != expected_size:
        raise ValueError(f"declared image size {expected_size} != decoded size {image.size}")
    left, top, right, bottom = candidate["bbox"]
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("crop bounds exceed decoded image geometry")
    return image.crop((left, top, right, bottom))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--dataset-version", default="1")
    parser.add_argument("--min-side", type=int, default=24)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    actual_md5 = file_md5(args.archive)
    if actual_md5.lower() != args.expected_md5.lower():
        raise RuntimeError(f"archive MD5 mismatch {actual_md5} != {args.expected_md5}")
    archive_sha = file_sha256(args.archive)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"output root is not empty; refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    crops_root = args.output_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    raw_labels = Counter()
    raw_types = Counter()
    raw_colors = Counter()
    candidates = []
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if any(not safe_zip_name(name) for name in names):
            raise RuntimeError("archive contains an unsafe member path")
        images = paired_images(names)
        xml_names = sorted(name for name in names if PurePosixPath(name).suffix.lower() == ".xml")
        counters["archive_entries"] = len(names)
        counters["image_files"] = len(images)
        counters["xml_files"] = len(xml_names)
        for xml_name in xml_names:
            root = ET.fromstring(archive.read(xml_name))
            stem = str(PurePosixPath(xml_name).with_suffix(""))
            image_name = images.get(stem)
            if not image_name:
                counters["xml_missing_paired_image"] += 1
                continue
            declared_width = int(float(root.findtext("size/width", "0") or 0))
            declared_height = int(float(root.findtext("size/height", "0") or 0))
            group = contributor_group(image_name)
            sequence_group = capture_sequence_group(image_name)
            for object_index, node in enumerate(root.findall("object")):
                counters["objects"] += 1
                raw_label = node.findtext("name", "").strip().lower()
                raw_labels[raw_label] += 1
                color, vehicle_type = parse_label(raw_label)
                raw_types[vehicle_type] += 1
                raw_colors[color] += 1
                if color == "unknown":
                    counters["rejected_unknown_or_ambiguous_color"] += 1
                    continue
                if vehicle_type in EXCLUDED_TYPES:
                    counters[f"rejected_non_four_wheel_{vehicle_type}"] += 1
                    continue
                if vehicle_type not in ACCEPTED_TYPES:
                    counters["rejected_unknown_vehicle_type"] += 1
                    continue
                if bool_xml(node, "truncated") or bool_xml(node, "occluded") or bool_xml(node, "difficult"):
                    counters["rejected_visibility_flag"] += 1
                    continue
                box = node.find("bndbox")
                if box is None:
                    counters["rejected_missing_bbox"] += 1
                    continue
                xmin = float(box.findtext("xmin", "nan")); ymin = float(box.findtext("ymin", "nan"))
                xmax = float(box.findtext("xmax", "nan")); ymax = float(box.findtext("ymax", "nan"))
                if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
                    counters["rejected_nonfinite_bbox"] += 1
                    continue
                left = max(0, math.floor(xmin)); top = max(0, math.floor(ymin))
                right = min(declared_width, math.ceil(xmax)); bottom = min(declared_height, math.ceil(ymax))
                if declared_width <= 0 or declared_height <= 0 or right <= left or bottom <= top:
                    counters["rejected_invalid_bbox"] += 1
                    continue
                candidates.append({
                    "xml_zip_path": xml_name, "image_zip_path": image_name,
                    "object_index": object_index, "raw_label": raw_label,
                    "color": color, "vehicle_type": vehicle_type, "source_group": group,
                    "sequence_group": sequence_group,
                    "declared_width": declared_width, "declared_height": declared_height,
                    "bbox": (left, top, right, bottom),
                })

        # First image pass: validate declared geometry and collect quality/dedup fingerprints.
        valid = []
        for candidate in candidates:
            try:
                crop = load_crop(archive, candidate)
                width, height = crop.size
                if min(width, height) < args.min_side:
                    counters["rejected_too_small"] += 1
                    continue
                mean, contrast, edge_variance = quality(crop)
                if mean < 10.0 or mean > 248.0 or contrast < 8.0 or edge_variance < 4.0:
                    counters["rejected_color_visibility_quality"] += 1
                    continue
                candidate.update({
                    "width": width, "height": height, "mean": mean, "contrast": contrast,
                    "edge_variance": edge_variance, "normalized_sha256": normalized_sha(crop),
                    "dhash64": dhash64(crop), "quality_score": contrast + math.log1p(edge_variance) * 5.0,
                })
                valid.append(candidate)
            except Exception:
                counters["rejected_unreadable_image_or_crop"] += 1

        # Exact duplicates are global so contradictory reused crops fail closed.
        # Perceptual duplicates are limited to the same capture burst and raw type,
        # preventing similar but distinct vehicles from collapsing.
        uf = UnionFind(len(valid))
        exact_by_hash: dict[str, list[int]] = defaultdict(list)
        sequence_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, item in enumerate(valid):
            exact_by_hash[item["normalized_sha256"]].append(index)
            sequence_indexes[(item["sequence_group"], item["vehicle_type"])].append(index)
        for indexes in exact_by_hash.values():
            for index in indexes[1:]:
                uf.union(indexes[0], index)
        for indexes in sequence_indexes.values():
            for position, left in enumerate(indexes):
                for right in indexes[position + 1:]:
                    if (valid[left]["dhash64"] ^ valid[right]["dhash64"]).bit_count() <= args.near_duplicate_hamming:
                        uf.union(left, right)
        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(valid)):
            groups[uf.find(index)].append(index)
        selected_indexes = []
        for indexes in groups.values():
            colors = {valid[index]["color"] for index in indexes}
            if len(colors) != 1:
                counters["rejected_conflicting_duplicate_group"] += len(indexes)
                continue
            selected = max(indexes, key=lambda index: valid[index]["quality_score"])
            selected_indexes.append(selected)
            counters["deduplicated_same_color_rows"] += len(indexes) - 1

        selected_items = [valid[index] for index in sorted(selected_indexes)]
        color_counts = Counter(item["color"] for item in selected_items)
        inverse_sqrt = {color: 1.0 / math.sqrt(count) for color, count in color_counts.items()}
        mean_weight = sum(inverse_sqrt.values()) / max(1, len(inverse_sqrt))
        rows = []
        for item_index, item in enumerate(selected_items):
            crop = load_crop(archive, item)
            name = f"kvc_{item_index:05d}_{item['color']}.jpg"
            output_path = crops_root / name
            crop.save(output_path, format="JPEG", quality=95, optimize=True)
            output_sha256 = file_sha256(output_path)
            mean, contrast, edge_variance = item["mean"], item["contrast"], item["edge_variance"]
            low_light = mean < 70.0
            small = min(item["width"], item["height"]) < 48 or item["width"] * item["height"] < 4096
            sample_weight = min(3.0, max(1.0, inverse_sqrt[item["color"]] / mean_weight))
            rows.append({
                "image_path": f"kaggle-color-cc0-v1/crops/{name}",
                "body_type": "unknown", "color": item["color"],
                "crop_quality": "approved", "viewpoint": "unknown",
                "blur": str(edge_variance < 20.0).lower(), "occluded": "false", "truncated": "false",
                "night": str(low_light).lower(), "camera_id": "", "video_id": item["source_group"],
                "track_group": f"KVC:{item['source_group']}", "split": "train",
                "source_frame_id": item["image_zip_path"], "review_status": "approved",
                "body_type_supervised": "false", "color_supervised": "true",
                "annotation_source": "manual_pascal_voc", "source_dataset": "Kaggle-DataCluster-Vehicle-Color-CC0",
                "source_manifest": str(args.archive), "source_license": "CC0-1.0",
                "review_method": "source_manual_review_plus_fail_closed_geometry_quality_dedup",
                "sha256": output_sha256, "dhash64": f"{item['dhash64']:016x}",
                "occlusion_level": "visible", "vehicle_size": "small" if small else "medium",
                "lighting": "low_light" if low_light else "daylight", "label_confidence": "high",
                "license_train_eligible": "true", "source_group": item["source_group"],
                "color_review_status": "approved", "review_score": "1.0", "review_model": "none",
                "formal_train_eligible": "true", "hard_example_priority": "1" if low_light or small else "0",
                "hard_mining_tags": ";".join(tag for tag, enabled in (("low_light", low_light), ("small_target", small)) if enabled) or "normal",
                "hard_score": f"{float(low_light) + float(small):.3f}",
                "photometric_mean": f"{mean:.4f}", "photometric_contrast": f"{contrast:.4f}",
                "photometric_sharpness": f"{edge_variance:.4f}", "pseudo_label": "false",
                "pseudo_label_confidence": "", "source_image_id": item["image_zip_path"],
                "small_target": str(int(small)), "low_light": str(int(low_light)),
                "soft": str(int(edge_variance < 20.0)), "crop_mean_luma": f"{mean:.4f}",
                "crop_sharpness": f"{edge_variance:.4f}", "crop_sha256": output_sha256,
                "sample_weight": f"{sample_weight:.6f}",
                "training_reason": "CC0 explicit manual color label; four-wheel visible box; quality and dedup passed",
            })

    manifest_path = args.output_root / "attribute_supplement.csv"
    fields = list(rows[0]) if rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "kaggle-vehicle-color-cc0-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if len(rows) >= 100 else "fail_closed_insufficient_accepted_rows",
        "source": {
            "dataset_ref": "dataclusterlabs/vehicle-color-detection-dataset",
            "dataset_version": args.dataset_version, "license": "CC0-1.0",
            "archive": str(args.archive.resolve()), "archive_bytes": args.archive.stat().st_size,
            "archive_md5": actual_md5, "archive_sha256": archive_sha,
        },
        "policy": {
            "training_eligible": True, "split": "train_only_supplement",
            "accepted_vehicle_types": sorted(ACCEPTED_TYPES),
            "excluded_vehicle_types": sorted(EXCLUDED_TYPES),
            "model_predictions_used": False, "frozen_video_used": False,
            "visibility": "truncated, occluded and difficult boxes rejected",
            "dedup": f"global exact normalized RGB plus capture-burst/type-local dHash <= {args.near_duplicate_hamming}; conflicts reject entire group",
        },
        "counters": dict(sorted(counters.items())),
        "raw_object_label_counts": dict(raw_labels.most_common()),
        "raw_vehicle_type_counts": dict(raw_types.most_common()),
        "raw_mapped_color_counts": dict(raw_colors.most_common()),
        "accepted_rows": len(rows),
        "accepted_color_counts": dict(sorted(Counter(row["color"] for row in rows).items())),
        "accepted_low_light_rows": sum(row["low_light"] == "1" for row in rows),
        "accepted_small_rows": sum(row["small_target"] == "1" for row in rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "decision": "append only approved rows to the Stage48 train split; preserve all existing validation/test rows unchanged",
    }
    report_path = args.output_root / "audit-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
