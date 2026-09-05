#!/usr/bin/env python3
"""Re-audit train-only scene metadata and group-aware duplicates.

The source manifest is immutable. Validation/test pixels are never opened. The
script writes an enriched copy plus a compact overlay and an evidence report.
Machine photometry is deliberately reported as a low-light proxy; it never
upgrades a sample to source-confirmed night.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


TRUE_VALUES = {"1", "true", "yes", "y"}
NIGHT_VALUES = {"night", "nighttime", "night_source_truth"}
LOWLIGHT_VALUES = {"low_light", "lowlight"}
DAY_VALUES = {"day", "daylight", "daytime", "moderate_light"}
FROZEN_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}
OVERLAY_FIELDS = [
    "stage177_row_number", "image_path", "source_dataset", "camera_id",
    "video_id", "track_group", "source_group", "body_type", "color",
    "stage177_group_key", "stage177_scene_group_key", "stage177_pixel_sha256",
    "stage177_dhash64", "stage177_mean_luma", "stage177_median_luma",
    "stage177_border_mean_luma", "stage177_dark_fraction",
    "stage177_highlight_fraction", "stage177_contrast", "stage177_sharpness",
    "stage177_lowlight_score", "stage177_frame_lowlight_proxy",
    "stage177_frame_night_candidate", "stage177_scene_label",
    "stage177_scene_origin", "stage177_scene_confidence",
    "stage177_machine_night_candidate", "stage177_scene_conflict",
    "stage177_exact_duplicate_of", "stage177_near_duplicate_of",
    "stage177_label_conflict", "stage177_dedup_eligible",
    "stage177_effective_representative", "stage177_read_error",
]


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def supervised(row: dict[str, str], head: str) -> bool:
    value = str(row.get(f"{head}_supervised") or "").strip()
    return True if not value else truthy(value)


def eligible_train(
    row: dict[str, str],
    body_types: set[str],
    colors: set[str],
    *,
    include_coarse_body: bool = True,
    include_coarse_color: bool = False,
) -> bool:
    body = str(row.get("body_type") or "").strip().lower()
    color = str(row.get("color") or "").strip().lower()
    coarse_body = str(row.get("coarse_body_family") or "").strip().lower()
    coarse_color = str(row.get("coarse_color_group") or "").strip().lower()
    return (
        str(row.get("split") or "").strip().lower() == "train"
        and str(row.get("review_status") or "").strip().lower() == "approved"
        and (
            (supervised(row, "body_type") and body in body_types)
            or (supervised(row, "color") and color in colors)
            or (include_coarse_body and coarse_body in {"car", "truck"})
            or (include_coarse_color and coarse_color in {"1", "2", "3"})
        )
    )


def audit_scope(
    row: dict[str, str],
    body_types: set[str],
    colors: set[str],
    *,
    include_all_train: bool = False,
    include_coarse_body: bool = True,
    include_coarse_color: bool = False,
) -> bool:
    if include_all_train:
        return str(row.get("split") or "").strip().lower() == "train"
    return eligible_train(
        row,
        body_types,
        colors,
        include_coarse_body=include_coarse_body,
        include_coarse_color=include_coarse_color,
    )


def label_signature(row: dict[str, str]) -> tuple[str, str]:
    body = str(row.get("body_type") or "").strip().lower() if supervised(row, "body_type") else ""
    color = str(row.get("color") or "").strip().lower() if supervised(row, "color") else ""
    return ("" if body in {"", "unknown"} else body, "" if color in {"", "unknown"} else color)


def signatures_conflict(left: tuple[str, str], right: tuple[str, str]) -> bool:
    return any(a and b and a != b for a, b in zip(left, right))


def exact_representative_key(record: dict[str, Any]) -> tuple[int, int, int, str, int]:
    supervised_heads = int(bool(record["body_exact_supervised"])) + int(
        bool(record["color_exact_supervised"])
    )
    return (
        -supervised_heads,
        -int(record["explicit_label"] != "unknown"),
        -int(bool(record["track_group"])),
        record["image_path"],
        record["row_number"],
    )


def group_keys(row: dict[str, str]) -> tuple[str, str, bool]:
    source = str(row.get("source_dataset") or row.get("source_group") or "unknown").strip()
    camera = str(row.get("camera_id") or "").strip()
    video = str(row.get("video_id") or "").strip()
    track = str(row.get("track_group") or "").strip()
    source_group = str(row.get("source_group") or "").strip()
    source_image = str(row.get("source_image_id") or row.get("source_frame_id") or "").strip()
    if track:
        dedup = "|".join((source, camera, video, "track", track))
        scene = "|".join((source, camera, video or track))
        return dedup, scene, True
    if video:
        key = "|".join((source, camera, "video", video))
        return key, "|".join((source, camera, video)), False
    if source_group:
        return "|".join((source, "source_group", source_group)), "", False
    if source_image:
        return "|".join((source, "source_image", source_image)), "", False
    path = str(row.get("image_path") or "").strip()
    return "|".join((source, "path", path)), "", False


def explicit_scene(row: dict[str, str]) -> tuple[str, str, float]:
    lighting = str(row.get("lighting") or "").strip().lower()
    if truthy(row.get("night")) or lighting in NIGHT_VALUES:
        return "night", "existing_explicit_night", 1.0
    if truthy(row.get("low_light")) or lighting in LOWLIGHT_VALUES:
        return "low_light", "existing_explicit_low_light", 0.95
    if lighting in DAY_VALUES:
        return "daylight", "existing_explicit_daylight", 0.95
    return "unknown", "missing_scene_metadata", 0.0


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def analyze_image(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    key, path_string = item
    try:
        payload = Path(path_string).read_bytes()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("decode_failed")
        height, width = image.shape[:2]
        scale = min(1.0, 192.0 / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32)
        height, width = gray.shape
        border_y = max(2, round(height * 0.12))
        border_x = max(2, round(width * 0.12))
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
        highlight_fraction = float((gray > 235.0).mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        score = (
            0.28 * float(np.clip((95.0 - mean) / 65.0, 0.0, 1.0))
            + 0.22 * float(np.clip((90.0 - median) / 65.0, 0.0, 1.0))
            + 0.25 * float(np.clip((100.0 - border_mean) / 70.0, 0.0, 1.0))
            + 0.15 * min(1.0, dark_fraction / 0.60)
            + 0.10 * float(np.clip((165.0 - p90) / 105.0, 0.0, 1.0))
        )
        frame_lowlight = (
            score >= 0.62 and mean <= 88.0 and median <= 82.0
            and border_mean <= 95.0 and dark_fraction >= 0.30
        )
        frame_night_candidate = (
            score >= 0.78 and mean <= 70.0 and median <= 65.0
            and border_mean <= 80.0 and dark_fraction >= 0.45
        )
        bright = score <= 0.15 and mean >= 130.0 and median >= 120.0 and dark_fraction <= 0.10
        return key, {
            "pixel_sha256": hashlib.sha256(payload).hexdigest(),
            "dhash64": dhash64(gray_u8),
            "mean_luma": mean, "median_luma": median,
            "border_mean_luma": border_mean, "dark_fraction": dark_fraction,
            "highlight_fraction": highlight_fraction, "contrast": contrast,
            "sharpness": sharpness, "lowlight_score": score,
            "frame_lowlight": frame_lowlight,
            "frame_night_candidate": frame_night_candidate,
            "bright": bright, "error": "",
        }
    except Exception as error:  # failure-close: unreadable images stay ineligible
        return key, {"error": f"{type(error).__name__}:{error}"}


class NearDuplicateIndex:
    def __init__(self, maximum_distance: int) -> None:
        self.maximum_distance = maximum_distance
        self.bands: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def match(self, value: int) -> int | None:
        candidates: set[tuple[int, int]] = set()
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            candidates.update(self.bands.get((band_index, band), []))
        for existing, index in candidates:
            if (existing ^ value).bit_count() <= self.maximum_distance:
                return index
        return None

    def add(self, value: int, index: int) -> None:
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            self.bands[(band_index, band)].append((value, index))


def evenly_spaced(indexes: list[int], maximum: int) -> set[int]:
    if len(indexes) <= maximum:
        return set(indexes)
    if maximum == 1:
        return {indexes[len(indexes) // 2]}
    positions = {round(i * (len(indexes) - 1) / (maximum - 1)) for i in range(maximum)}
    return {indexes[position] for position in positions}


def scene_counts(indexes: Iterable[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(records[index]["stage177_scene_label"] for index in indexes)
    total = sum(labels.values())
    explicit_night = labels["night"]
    explicit_low = sum(
        1 for index in indexes
        if records[index]["stage177_scene_origin"] == "existing_explicit_low_light"
    )
    machine_track_low = sum(
        1 for index in indexes
        if str(records[index]["stage177_scene_origin"]).startswith("machine_track_")
        and records[index]["stage177_scene_label"] == "low_light"
    )
    machine_single_low = sum(
        1 for index in indexes
        if str(records[index]["stage177_scene_origin"]).startswith("machine_single_frame_")
        and records[index]["stage177_scene_label"] == "low_light"
    )
    adverse = explicit_night + explicit_low + machine_track_low
    return {
        "total": total,
        "scene_labels": dict(sorted(labels.items())),
        "existing_explicit_night": explicit_night,
        "existing_explicit_low_light": explicit_low,
        "machine_track_low_light_proxy": machine_track_low,
        "machine_single_frame_low_light_candidate_excluded_from_quota": machine_single_low,
        "effective_adverse_light": adverse,
        "confirmed_night_fraction": explicit_night / total if total else 0.0,
        "effective_adverse_light_fraction": adverse / total if total else 0.0,
        "unknown_fraction": labels["unknown"] / total if total else 0.0,
    }


def attribute_supervision_counts(
    indexes: Iterable[int], records: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = list(indexes)
    total = len(selected)
    night = [index for index in selected if records[index]["stage177_scene_label"] == "night"]
    low_light = [index for index in selected if records[index]["stage177_scene_label"] == "low_light"]

    def count(flag: str, members: list[int]) -> int:
        return sum(bool(records[index][flag]) for index in members)

    body_total = count("body_exact_supervised", selected)
    color_total = count("color_exact_supervised", selected)
    body_night = count("body_exact_supervised", night)
    color_night = count("color_exact_supervised", night)
    body_low_light = count("body_exact_supervised", low_light)
    color_low_light = count("color_exact_supervised", low_light)
    return {
        "total": total,
        "confirmed_night_rows": len(night),
        "low_light_rows": len(low_light),
        "exact_body_supervised_rows": body_total,
        "exact_color_supervised_rows": color_total,
        "confirmed_night_exact_body_supervised_rows": body_night,
        "confirmed_night_exact_color_supervised_rows": color_night,
        "low_light_exact_body_supervised_rows": body_low_light,
        "low_light_exact_color_supervised_rows": color_low_light,
        "confirmed_night_exact_body_supervised_fraction_of_all": body_night / total if total else 0.0,
        "confirmed_night_exact_color_supervised_fraction_of_all": color_night / total if total else 0.0,
        "exact_body_supervision_rate_within_confirmed_night": body_night / len(night) if night else 0.0,
        "exact_color_supervision_rate_within_confirmed_night": color_night / len(night) if night else 0.0,
    }


def grouping_counts(indexes: Iterable[int], records: list[dict[str, Any]]) -> dict[str, int]:
    selected = [records[index] for index in indexes]
    sources = {record["source_dataset"] for record in selected if record["source_dataset"]}
    cameras = {
        (record["source_dataset"], record["camera_id"])
        for record in selected if record["camera_id"]
    }
    videos = {
        (record["source_dataset"], record["camera_id"], record["video_id"])
        for record in selected if record["video_id"]
    }
    tracks = {
        (record["source_dataset"], record["camera_id"], record["video_id"], record["track_group"])
        for record in selected if record["track_group"]
    }
    source_groups = {
        (record["source_dataset"], record["source_group"])
        for record in selected if record["source_group"]
    }
    return {
        "rows": len(selected),
        "distinct_sources": len(sources),
        "distinct_cameras": len(cameras),
        "distinct_videos": len(videos),
        "distinct_tracks": len(tracks),
        "distinct_source_groups": len(source_groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--max-effective-frames-per-track", type=int, default=5)
    parser.add_argument("--include-all-train", action="store_true")
    parser.add_argument("--include-coarse-body", action="store_true")
    parser.add_argument("--include-coarse-color", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    allowed_root = args.allowed_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    actual_manifest_sha = sha256_file(manifest)
    if actual_manifest_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_manifest_sha}")
    labels = args.labels.resolve()
    actual_labels_sha = sha256_file(labels)
    if actual_labels_sha.lower() != args.expected_labels_sha256.lower():
        raise RuntimeError(f"labels SHA256 mismatch: {actual_labels_sha}")
    label_data = json.loads(labels.read_text(encoding="utf-8"))
    body_types = {str(value).strip().lower() for value in label_data["body_types"]}
    colors = {str(value).strip().lower() for value in label_data["colors"]}
    body_types.discard("unknown")
    colors.discard("unknown")

    records: list[dict[str, Any]] = []
    unique_paths: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    missing_scene_metadata = 0
    formal_train_eligible_rows = 0
    frozen_rows: list[int] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        required = {"image_path", "split", "review_status", "body_type", "color"}
        if missing := required - set(original_fields):
            raise RuntimeError(f"manifest missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            split_counts[str(row.get("split") or "")] += 1
            searchable = " ".join(str(row.get(field) or "") for field in (
                "image_path", "source_manifest", "source_frame_id", "video_id",
            )).lower()
            if any(marker in searchable for marker in FROZEN_MARKERS):
                frozen_rows.append(row_number)
            if not audit_scope(
                row, body_types, colors,
                include_all_train=args.include_all_train,
                include_coarse_body=args.include_coarse_body,
                include_coarse_color=args.include_coarse_color,
            ):
                continue
            formal_train_eligible_rows += int(truthy(row.get("formal_train_eligible")))
            path = Path(str(row.get("image_path") or ""))
            absolute = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
            try:
                absolute.relative_to(allowed_root)
            except ValueError as error:
                raise RuntimeError(f"train image outside allowed root at row {row_number}: {absolute}") from error
            explicit_label, explicit_origin, explicit_confidence = explicit_scene(row)
            missing_scene_metadata += int(explicit_label == "unknown")
            dedup_key, scene_key, track_bounded = group_keys(row)
            record = {
                "row_number": row_number,
                "image_path": str(row.get("image_path") or ""),
                "absolute_path": str(absolute),
                "source_dataset": str(row.get("source_dataset") or "unknown"),
                "camera_id": str(row.get("camera_id") or ""),
                "video_id": str(row.get("video_id") or ""),
                "track_group": str(row.get("track_group") or ""),
                "source_group": str(row.get("source_group") or ""),
                "input_role": str(row.get("stage214_input_role") or "single_manifest"),
                "source_frame_id": str(row.get("source_frame_id") or ""),
                "body_type": str(row.get("body_type") or ""),
                "color": str(row.get("color") or ""),
                "body_exact_supervised": (
                    supervised(row, "body_type")
                    and str(row.get("body_type") or "").strip().lower() in body_types
                ),
                "color_exact_supervised": (
                    supervised(row, "color")
                    and str(row.get("color") or "").strip().lower() in colors
                ),
                "label_signature": label_signature(row),
                "group_key": dedup_key, "scene_group_key": scene_key,
                "track_bounded": track_bounded,
                "explicit_label": explicit_label, "explicit_origin": explicit_origin,
                "explicit_confidence": explicit_confidence,
            }
            records.append(record)
            unique_paths.setdefault(record["image_path"], record["absolute_path"])
    if frozen_rows:
        raise RuntimeError(f"frozen-video marker found in rows {frozen_rows[:20]}")
    if not records:
        raise RuntimeError("no train rows in the selected audit scope")

    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "stage177-state.json"
    state: dict[str, Any] = {
        "schema_version": "stage177-full-train-scene-audit-state-v1",
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest), "manifest_sha256": actual_manifest_sha,
        "labels": str(labels), "labels_sha256": actual_labels_sha,
        "eligible_train_rows": len(records), "unique_image_paths": len(unique_paths),
        "unique_images_scanned": 0, "validation_or_test_images_opened": 0,
        "frozen_video_used": False,
    }
    atomic_json(state_path, state)

    pixel_by_path: dict[str, dict[str, Any]] = {}
    items = list(unique_paths.items())
    workers = max(1, min(args.workers, 16))
    batch_size = max(workers, args.batch_size)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(items), batch_size):
            batch = items[offset: offset + batch_size]
            futures = [executor.submit(analyze_image, item) for item in batch]
            for future in as_completed(futures):
                path_key, metrics = future.result()
                pixel_by_path[path_key] = metrics
            completed = min(len(items), offset + len(batch))
            state.update(
                updated_at=datetime.now(timezone.utc).isoformat(),
                unique_images_scanned=completed,
                progress=completed / len(items),
            )
            atomic_json(state_path, state)
            print(json.dumps({"scanned": completed, "total": len(items), "progress": state["progress"]}), flush=True)

    unreadable = 0
    for record in records:
        metrics = pixel_by_path[record["image_path"]]
        record["metrics"] = metrics
        unreadable += int(bool(metrics.get("error")))
        record.update({
            "exact_duplicate_of": "", "near_duplicate_of": "",
            "label_conflict": False, "dedup_eligible": not bool(metrics.get("error")),
            "effective_representative": False, "scene_conflict": False,
            "machine_night_candidate": False,
        })

    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pixel_sha = str(record["metrics"].get("pixel_sha256") or "")
        if pixel_sha:
            exact_groups[pixel_sha].append(index)
    exact_duplicates = 0
    exact_conflict_rows = 0
    exact_kept: list[int] = []
    for indexes in exact_groups.values():
        bodies = {records[index]["label_signature"][0] for index in indexes if records[index]["label_signature"][0]}
        colors = {records[index]["label_signature"][1] for index in indexes if records[index]["label_signature"][1]}
        conflict = len(bodies) > 1 or len(colors) > 1
        if conflict:
            exact_conflict_rows += len(indexes)
            for index in indexes:
                records[index]["label_conflict"] = True
                records[index]["dedup_eligible"] = False
            continue
        indexes.sort(key=lambda index: exact_representative_key(records[index]))
        representative = indexes[0]
        records[representative]["body_exact_supervised"] = any(
            records[index]["body_exact_supervised"] for index in indexes
        )
        records[representative]["color_exact_supervised"] = any(
            records[index]["color_exact_supervised"] for index in indexes
        )
        exact_kept.append(representative)
        for duplicate in indexes[1:]:
            records[duplicate]["exact_duplicate_of"] = records[representative]["image_path"]
            records[duplicate]["dedup_eligible"] = False
            exact_duplicates += 1

    near_groups: dict[str, list[int]] = defaultdict(list)
    for index in exact_kept:
        if records[index]["dedup_eligible"]:
            near_groups[records[index]["group_key"]].append(index)
    near_duplicates = 0
    near_conflict_rows: set[int] = set()
    for indexes in near_groups.values():
        indexes.sort(key=lambda index: (records[index]["source_frame_id"], records[index]["image_path"]))
        tree = NearDuplicateIndex(args.near_duplicate_hamming)
        for index in indexes:
            value = int(records[index]["metrics"]["dhash64"])
            matched = tree.match(value)
            if matched is None:
                tree.add(value, index)
                continue
            if signatures_conflict(records[index]["label_signature"], records[matched]["label_signature"]):
                records[index]["label_conflict"] = True
                records[matched]["label_conflict"] = True
                records[index]["dedup_eligible"] = False
                records[matched]["dedup_eligible"] = False
                near_conflict_rows.update((index, matched))
            else:
                records[index]["near_duplicate_of"] = records[matched]["image_path"]
                records[index]["dedup_eligible"] = False
                near_duplicates += 1

    scene_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["dedup_eligible"] and record["scene_group_key"]:
            scene_groups[record["scene_group_key"]].append(index)
    scene_consensus: dict[str, tuple[bool, bool, float, int]] = {}
    for key, indexes in scene_groups.items():
        metrics = [records[index]["metrics"] for index in indexes]
        scores = sorted(float(item["lowlight_score"]) for item in metrics)
        low_fraction = sum(bool(item["frame_lowlight"]) for item in metrics) / len(metrics)
        night_fraction = sum(bool(item["frame_night_candidate"]) for item in metrics) / len(metrics)
        median_score = float(np.median(scores))
        low_consensus = len(indexes) >= 3 and low_fraction >= 0.80 and median_score >= 0.65
        night_consensus = len(indexes) >= 3 and night_fraction >= 0.80 and median_score >= 0.78
        scene_consensus[key] = (low_consensus, night_consensus, median_score, len(indexes))

    for record in records:
        metrics = record["metrics"]
        label, origin, confidence = record["explicit_label"], record["explicit_origin"], record["explicit_confidence"]
        if metrics.get("error"):
            label, origin, confidence = "unknown", "unreadable_failure_close", 0.0
        else:
            low_consensus, night_consensus, median_score, group_size = scene_consensus.get(
                record["scene_group_key"], (False, False, 0.0, 0),
            )
            record["machine_night_candidate"] = bool(night_consensus or metrics["frame_night_candidate"])
            if label == "daylight" and metrics["frame_lowlight"]:
                record["scene_conflict"] = True
            if label == "unknown" and night_consensus:
                label, origin = "low_light", "machine_track_night_candidate"
                confidence = min(0.99, 0.80 + 0.19 * median_score)
            elif label == "unknown" and low_consensus:
                label, origin = "low_light", "machine_track_lowlight_consensus"
                confidence = min(0.97, 0.75 + 0.20 * median_score)
            elif label == "unknown" and metrics["frame_night_candidate"]:
                label, origin = "low_light", "machine_single_frame_night_candidate"
                confidence = 0.80
            elif label == "unknown" and metrics["frame_lowlight"] and metrics["lowlight_score"] >= 0.72:
                label, origin = "low_light", "machine_single_frame_lowlight_proxy"
                confidence = 0.75
            elif label == "unknown" and metrics["bright"]:
                label, origin = "daylight", "machine_bright_proxy"
                confidence = 0.75
        record["stage177_scene_label"] = label
        record["stage177_scene_origin"] = origin
        record["stage177_scene_confidence"] = confidence

    dedup_indexes = [index for index, record in enumerate(records) if record["dedup_eligible"]]
    effective_indexes: set[int] = set()
    bounded_groups: dict[str, list[int]] = defaultdict(list)
    for index in dedup_indexes:
        record = records[index]
        if record["track_bounded"]:
            bounded_groups[record["group_key"]].append(index)
        else:
            effective_indexes.add(index)
    for indexes in bounded_groups.values():
        indexes.sort(key=lambda index: (records[index]["source_frame_id"], records[index]["image_path"]))
        effective_indexes.update(evenly_spaced(indexes, args.max_effective_frames_per_track))
    for index in effective_indexes:
        records[index]["effective_representative"] = True

    overlay_path = args.output_root / "stage177-scene-overlay.csv"
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERLAY_FIELDS)
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            def number(name: str) -> str:
                return "" if name not in metrics else f"{float(metrics[name]):.8f}"
            writer.writerow({
                "stage177_row_number": record["row_number"], "image_path": record["image_path"],
                "source_dataset": record["source_dataset"], "camera_id": record["camera_id"],
                "video_id": record["video_id"], "track_group": record["track_group"],
                "source_group": record["source_group"], "body_type": record["body_type"],
                "color": record["color"], "stage177_group_key": record["group_key"],
                "stage177_scene_group_key": record["scene_group_key"],
                "stage177_pixel_sha256": metrics.get("pixel_sha256", ""),
                "stage177_dhash64": "" if "dhash64" not in metrics else f"{int(metrics['dhash64']):016x}",
                "stage177_mean_luma": number("mean_luma"), "stage177_median_luma": number("median_luma"),
                "stage177_border_mean_luma": number("border_mean_luma"),
                "stage177_dark_fraction": number("dark_fraction"),
                "stage177_highlight_fraction": number("highlight_fraction"),
                "stage177_contrast": number("contrast"), "stage177_sharpness": number("sharpness"),
                "stage177_lowlight_score": number("lowlight_score"),
                "stage177_frame_lowlight_proxy": str(bool(metrics.get("frame_lowlight"))).lower(),
                "stage177_frame_night_candidate": str(bool(metrics.get("frame_night_candidate"))).lower(),
                "stage177_scene_label": record["stage177_scene_label"],
                "stage177_scene_origin": record["stage177_scene_origin"],
                "stage177_scene_confidence": f"{float(record['stage177_scene_confidence']):.6f}",
                "stage177_machine_night_candidate": str(record["machine_night_candidate"]).lower(),
                "stage177_scene_conflict": str(record["scene_conflict"]).lower(),
                "stage177_exact_duplicate_of": record["exact_duplicate_of"],
                "stage177_near_duplicate_of": record["near_duplicate_of"],
                "stage177_label_conflict": str(record["label_conflict"]).lower(),
                "stage177_dedup_eligible": str(record["dedup_eligible"]).lower(),
                "stage177_effective_representative": str(record["effective_representative"]).lower(),
                "stage177_read_error": metrics.get("error", ""),
            })

    overlay_by_row = {record["row_number"]: record for record in records}
    enriched_path = args.output_root / "attribute_manifest.stage177-scene-audited.csv"
    extra_fields = [field for field in OVERLAY_FIELDS if field not in {"stage177_row_number", "image_path", "source_dataset", "camera_id", "video_id", "track_group", "source_group", "body_type", "color"}]
    with manifest.open("r", encoding="utf-8-sig", newline="") as source, enriched_path.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=list(reader.fieldnames or []) + extra_fields)
        writer.writeheader()
        for row_number, row in enumerate(reader, 2):
            record = overlay_by_row.get(row_number)
            if record is not None:
                metrics = record["metrics"]
                values = {
                    "stage177_group_key": record["group_key"],
                    "stage177_scene_group_key": record["scene_group_key"],
                    "stage177_pixel_sha256": metrics.get("pixel_sha256", ""),
                    "stage177_dhash64": "" if "dhash64" not in metrics else f"{int(metrics['dhash64']):016x}",
                    "stage177_mean_luma": "" if "mean_luma" not in metrics else f"{float(metrics['mean_luma']):.8f}",
                    "stage177_median_luma": "" if "median_luma" not in metrics else f"{float(metrics['median_luma']):.8f}",
                    "stage177_border_mean_luma": "" if "border_mean_luma" not in metrics else f"{float(metrics['border_mean_luma']):.8f}",
                    "stage177_dark_fraction": "" if "dark_fraction" not in metrics else f"{float(metrics['dark_fraction']):.8f}",
                    "stage177_highlight_fraction": "" if "highlight_fraction" not in metrics else f"{float(metrics['highlight_fraction']):.8f}",
                    "stage177_contrast": "" if "contrast" not in metrics else f"{float(metrics['contrast']):.8f}",
                    "stage177_sharpness": "" if "sharpness" not in metrics else f"{float(metrics['sharpness']):.8f}",
                    "stage177_lowlight_score": "" if "lowlight_score" not in metrics else f"{float(metrics['lowlight_score']):.8f}",
                    "stage177_frame_lowlight_proxy": str(bool(metrics.get("frame_lowlight"))).lower(),
                    "stage177_frame_night_candidate": str(bool(metrics.get("frame_night_candidate"))).lower(),
                    "stage177_scene_label": record["stage177_scene_label"],
                    "stage177_scene_origin": record["stage177_scene_origin"],
                    "stage177_scene_confidence": f"{float(record['stage177_scene_confidence']):.6f}",
                    "stage177_machine_night_candidate": str(record["machine_night_candidate"]).lower(),
                    "stage177_scene_conflict": str(record["scene_conflict"]).lower(),
                    "stage177_exact_duplicate_of": record["exact_duplicate_of"],
                    "stage177_near_duplicate_of": record["near_duplicate_of"],
                    "stage177_label_conflict": str(record["label_conflict"]).lower(),
                    "stage177_dedup_eligible": str(record["dedup_eligible"]).lower(),
                    "stage177_effective_representative": str(record["effective_representative"]).lower(),
                    "stage177_read_error": metrics.get("error", ""),
                }
                row.update(values)
            writer.writerow(row)

    raw_indexes = list(range(len(records)))
    source_effective: dict[str, list[int]] = defaultdict(list)
    for index in effective_indexes:
        source_effective[records[index]["source_dataset"]].append(index)
    input_role_raw: dict[str, list[int]] = defaultdict(list)
    input_role_dedup: dict[str, list[int]] = defaultdict(list)
    input_role_effective: dict[str, list[int]] = defaultdict(list)
    for index in raw_indexes:
        input_role_raw[records[index]["input_role"]].append(index)
    for index in dedup_indexes:
        input_role_dedup[records[index]["input_role"]].append(index)
    for index in effective_indexes:
        input_role_effective[records[index]["input_role"]].append(index)
    input_roles = sorted(set(input_role_raw) | set(input_role_dedup) | set(input_role_effective))
    scene_conflicts = sum(record["scene_conflict"] for record in records)
    report = {
        "schema_version": "stage177-full-train-scene-audit-v1",
        "status": "pass" if unreadable == 0 and not exact_conflict_rows and not near_conflict_rows else "pass_with_fail_closed_exclusions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest": str(manifest), "sha256": actual_manifest_sha,
            "labels": str(labels), "labels_sha256": actual_labels_sha,
        },
        "scope": {
            "manifest_rows": sum(split_counts.values()), "split_counts": dict(sorted(split_counts.items())),
            "loader_usable_train_rows": len(records), "unique_image_paths_opened": len(unique_paths),
            "formal_train_eligible_rows_within_loader_scope": formal_train_eligible_rows,
            "validation_or_test_images_opened": 0, "frozen_video_used": False,
            "missing_explicit_scene_metadata_before_audit": missing_scene_metadata,
            "include_all_train_rows": args.include_all_train,
            "include_coarse_body_loader_rows": args.include_coarse_body,
            "include_coarse_color_loader_rows": args.include_coarse_color,
        },
        "deduplication": {
            "global_exact_duplicate_rows_removed": exact_duplicates,
            "exact_label_conflict_rows_excluded": exact_conflict_rows,
            "within_source_video_track_dhash_distance": args.near_duplicate_hamming,
            "group_near_duplicate_rows_removed": near_duplicates,
            "near_duplicate_label_conflict_rows_excluded": len(near_conflict_rows),
            "unreadable_rows_excluded": unreadable,
            "dedup_eligible_unique_images": len(dedup_indexes),
            "effective_track_balanced_representatives": len(effective_indexes),
            "max_effective_frames_per_track": args.max_effective_frames_per_track,
        },
        "scene_totals": {
            "raw_eligible_rows": scene_counts(raw_indexes, records),
            "dedup_eligible_unique_images": scene_counts(dedup_indexes, records),
            "track_balanced_effective_representatives": scene_counts(sorted(effective_indexes), records),
        },
        "attribute_supervision_totals": {
            "raw_eligible_rows": attribute_supervision_counts(raw_indexes, records),
            "dedup_eligible_unique_images": attribute_supervision_counts(dedup_indexes, records),
            "track_balanced_effective_representatives": attribute_supervision_counts(
                sorted(effective_indexes), records
            ),
        },
        "grouping_totals": {
            "raw_eligible_rows": grouping_counts(raw_indexes, records),
            "dedup_eligible_unique_images": grouping_counts(dedup_indexes, records),
            "track_balanced_effective_representatives": grouping_counts(
                sorted(effective_indexes), records
            ),
        },
        "per_source_effective": {
            source: scene_counts(indexes, records) for source, indexes in sorted(source_effective.items())
        },
        "per_source_effective_attribute_supervision": {
            source: attribute_supervision_counts(indexes, records)
            for source, indexes in sorted(source_effective.items())
        },
        "per_input_role": {
            role: {
                "raw": {
                    "scene": scene_counts(input_role_raw.get(role, []), records),
                    "attribute_supervision": attribute_supervision_counts(
                        input_role_raw.get(role, []), records
                    ),
                },
                "dedup": {
                    "scene": scene_counts(input_role_dedup.get(role, []), records),
                    "attribute_supervision": attribute_supervision_counts(
                        input_role_dedup.get(role, []), records
                    ),
                },
                "effective": {
                    "scene": scene_counts(input_role_effective.get(role, []), records),
                    "attribute_supervision": attribute_supervision_counts(
                        input_role_effective.get(role, []), records
                    ),
                },
            }
            for role in input_roles
        },
        "diagnostics": {
            "scene_conflicts_kept_as_existing_explicit_label": scene_conflicts,
            "machine_night_candidate_is_not_source_confirmed_night": True,
            "single_dark_frame_never_counted_as_source_confirmed_night": True,
        },
        "thresholds": {
            "lowlight_score_min": 0.62, "night_candidate_score_min": 0.78,
            "track_minimum_frames": 3, "track_minimum_positive_fraction": 0.80,
            "near_duplicate_hamming": args.near_duplicate_hamming,
        },
        "outputs": {
            "overlay": str(overlay_path), "overlay_sha256": sha256_file(overlay_path),
            "enriched_manifest": str(enriched_path), "enriched_manifest_sha256": sha256_file(enriched_path),
        },
        "policy": {
            "source_manifest_immutable": True, "body_and_color_labels_unchanged": True,
            "train_pixels_only": True, "validation_or_test_pixels_opened": False,
            "frozen_video_used": False, "production_model_modified": False,
            "training_started": False, "deployment_performed": False,
            "machine_photometry_is_proxy_not_real_night_truth": True,
        },
    }
    report_path = args.output_root / "stage177-full-train-scene-audit.json"
    atomic_json(report_path, report)
    for path in (overlay_path, enriched_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    state.update(
        status="complete", updated_at=datetime.now(timezone.utc).isoformat(),
        unique_images_scanned=len(unique_paths), progress=1.0,
        report=str(report_path), report_sha256=sha256_file(report_path),
    )
    atomic_json(state_path, state)
    print(json.dumps({
        "status": report["status"], "eligible_rows": len(records),
        "dedup_eligible": len(dedup_indexes), "effective_representatives": len(effective_indexes),
        "scene_totals": report["scene_totals"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
