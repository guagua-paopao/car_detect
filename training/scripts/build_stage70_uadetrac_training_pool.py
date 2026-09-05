#!/usr/bin/env python3
"""Build a fail-closed, training-partition-only UA-DETRAC attribute pool.

The source XML has exact ``bus``/``van`` labels and a coarse ``car`` label, but
does not contain vehicle-colour truth.  This builder therefore never invents a
fine car subtype or a colour label.  It also has no test-partition arguments,
which makes accidental test ingestion impossible through the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")
SEQUENCE_RE = re.compile(r"MVI_\d+", re.IGNORECASE)
EXACT_BODY_MAP = {"bus": "bus", "van": "van"}
COARSE_BODY_MAP = {"car": "car"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

FIELDS = [
    "image_path", "body_type", "color", "crop_quality", "viewpoint", "blur",
    "occluded", "truncated", "night", "camera_id", "video_id", "track_group",
    "split", "source_frame_id", "review_status", "body_type_supervised",
    "color_supervised", "annotation_source", "source_dataset", "source_manifest",
    "source_license", "review_method", "sha256", "dhash64", "width", "height",
    "gray_mean", "gray_stddev", "edge_variance", "occlusion_level", "vehicle_size",
    "lighting", "label_confidence", "license_train_eligible", "source_group",
    "hard_example_priority", "hard_mining_tags", "hard_score", "pseudo_label",
    "pseudo_label_confidence", "frame_number", "window_id", "window_pos",
    "source_vehicle_type", "source_color", "weather", "truncation_ratio",
    "track_body_truth", "track_color_truth", "track_label_agreement",
    "coarse_body_family", "source_partition", "dedup_policy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_name(value: str) -> str | None:
    match = SEQUENCE_RE.search(value)
    return match.group(0).upper() if match else None


def frame_number(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else None


def stable_split(sequence: str) -> str:
    bucket = int(hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:8], 16) % 5
    return "validation" if bucket == 0 else "train"


def assert_training_only(path: Path, label: str) -> Path:
    resolved = path.resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    lowered = str(resolved).lower()
    if {"test", "testing"} & lowered_parts or "vcas_rtsp_demo_60s" in lowered:
        raise RuntimeError(f"{label} is not a training-only path: {resolved}")
    return resolved


def read_image(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_jpeg(path: Path, image: np.ndarray) -> bool:
    try:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except (OSError, ValueError):
        return False


def dhash64(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def contiguous_windows(observations: list[dict], length: int, maximum: int) -> list[list[dict]]:
    ordered = sorted(observations, key=lambda row: row["frame_number"])
    candidates: list[list[dict]] = []
    for index in range(0, len(ordered) - length + 1):
        window = ordered[index:index + length]
        frames = [row["frame_number"] for row in window]
        if all(right - left == 1 for left, right in zip(frames, frames[1:])):
            candidates.append(window)
    if len(candidates) > maximum:
        selected_indexes: list[int] = []
        for point in np.linspace(0, len(candidates) - 1, maximum):
            index = int(round(float(point)))
            if index not in selected_indexes:
                selected_indexes.append(index)
        candidates = [candidates[index] for index in selected_indexes]
    accepted: list[list[dict]] = []
    used_frames: set[int] = set()
    for window in candidates:
        frames = {row["frame_number"] for row in window}
        if frames & used_frames:
            continue
        accepted.append(window)
        used_frames.update(frames)
    return accepted


def source_type_consensus(track: list[dict], minimum_share: float) -> tuple[str, float]:
    values = [str(row["source_vehicle_type"]).strip().lower() for row in track]
    if not values:
        return "unknown", 0.0
    value, count = Counter(values).most_common(1)[0]
    share = count / len(values)
    return (value, share) if share >= minimum_share else ("unknown", share)


def parse_training(training_root: Path, xml_root: Path) -> tuple[list[dict], dict]:
    image_dirs: dict[str, list[Path]] = defaultdict(list)
    for path in training_root.rglob("*"):
        if path.is_dir():
            sequence = sequence_name(path.name)
            if sequence:
                image_dirs[sequence].append(path)
    rows: list[dict] = []
    missing_sequences: list[str] = []
    xml_paths = sorted(xml_root.rglob("*.xml"))
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        sequence = sequence_name(root.attrib.get("name", "") or xml_path.stem)
        if not sequence:
            continue
        image_index: dict[int, Path] = {}
        for directory in image_dirs.get(sequence, []):
            for image in directory.iterdir():
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                    number = frame_number(image)
                    if number is not None:
                        image_index.setdefault(number, image)
        if not image_index:
            missing_sequences.append(sequence)
            continue
        sequence_attribute = root.find("sequence_attribute")
        weather = "unknown"
        camera_state = "unknown"
        if sequence_attribute is not None:
            weather = str(sequence_attribute.attrib.get(
                "weather", sequence_attribute.attrib.get("sence_weather", "unknown")
            )).strip().lower()
            camera_state = str(sequence_attribute.attrib.get("camera_state", "unknown")).strip().lower()
        for frame in root.findall("frame"):
            try:
                number = int(frame.attrib["num"])
            except (KeyError, ValueError):
                continue
            image = image_index.get(number)
            target_list = frame.find("target_list")
            if image is None or target_list is None:
                continue
            for target in target_list.findall("target"):
                box = target.find("box")
                attribute = target.find("attribute")
                if box is None or attribute is None:
                    continue
                try:
                    row = {
                        "sequence": sequence,
                        "split": stable_split(sequence),
                        "camera_state": camera_state,
                        "weather": weather,
                        "frame_number": number,
                        "image": image,
                        "track_id": int(target.attrib["id"]),
                        "left": float(box.attrib["left"]),
                        "top": float(box.attrib["top"]),
                        "box_width": float(box.attrib["width"]),
                        "box_height": float(box.attrib["height"]),
                        "source_vehicle_type": str(attribute.attrib.get("vehicle_type", "unknown")).strip().lower(),
                        "source_color": str(attribute.attrib.get("color", "unknown")).strip().lower(),
                        "truncation_ratio": float(attribute.attrib.get("truncation_ratio", 0.0) or 0.0),
                        "source_xml": xml_path,
                    }
                except (KeyError, ValueError):
                    continue
                occlusion = target.find("occlusion")
                row["occluded"] = occlusion is not None and len(list(occlusion)) > 0
                rows.append(row)
    return rows, {
        "xml_files": len(xml_paths),
        "observations": len(rows),
        "sequences_with_images": len({row["sequence"] for row in rows}),
        "missing_image_sequences": missing_sequences,
    }


def deduplicate(rows: list[dict], threshold: int) -> tuple[list[dict], dict]:
    # Prefer clear, visible crops.  Eight 8-bit bands guarantee a candidate
    # lookup collision whenever Hamming distance is at most seven.
    ordered = sorted(
        rows,
        key=lambda row: (
            row["occluded"] == "true" or row["truncated"] == "true",
            -float(row["edge_variance"]),
            row["image_path"],
        ),
    )
    exact_seen: dict[str, dict] = {}
    bands: list[dict[str, list[dict]]] = [defaultdict(list) for _ in range(8)]
    kept: list[dict] = []
    counts = Counter()
    cross_split_near_duplicates: list[dict] = []
    for row in ordered:
        exact = row["sha256"]
        if exact in exact_seen:
            counts["exact_duplicate"] += 1
            if exact_seen[exact]["split"] != row["split"]:
                counts["cross_split_exact_duplicate"] += 1
            Path(row["_absolute_path"]).unlink(missing_ok=True)
            continue
        candidates: dict[str, dict] = {}
        value = row["dhash64"]
        for index in range(8):
            key = value[index * 2:(index + 1) * 2]
            for candidate in bands[index].get(key, []):
                candidates[candidate["image_path"]] = candidate
        near = next((candidate for candidate in candidates.values()
                     if hamming(value, candidate["dhash64"]) <= threshold), None)
        if near is not None:
            counts["perceptual_near_duplicate"] += 1
            if near["split"] != row["split"]:
                counts["cross_split_perceptual_near_duplicate"] += 1
                if len(cross_split_near_duplicates) < 100:
                    cross_split_near_duplicates.append({
                        "kept": near["image_path"], "removed": row["image_path"],
                        "distance": hamming(value, near["dhash64"]),
                    })
            Path(row["_absolute_path"]).unlink(missing_ok=True)
            continue
        exact_seen[exact] = row
        for index in range(8):
            bands[index][value[index * 2:(index + 1) * 2]].append(row)
        kept.append(row)
    for row in kept:
        row.pop("_absolute_path", None)
    return kept, {
        "counts": dict(sorted(counts.items())),
        "cross_split_examples": cross_split_near_duplicates,
    }


def cross_split_near_duplicate_examples(
    rows: list[dict], threshold: int, limit: int = 100
) -> list[dict]:
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    bands: list[dict[str, list[dict]]] = [defaultdict(list) for _ in range(8)]
    for row in train_rows:
        value = row["dhash64"]
        for index in range(8):
            bands[index][value[index * 2:(index + 1) * 2]].append(row)
    examples: list[dict] = []
    for row in validation_rows:
        value = row["dhash64"]
        candidates: dict[str, dict] = {}
        for index in range(8):
            key = value[index * 2:(index + 1) * 2]
            for candidate in bands[index].get(key, []):
                candidates[candidate["image_path"]] = candidate
        for candidate in candidates.values():
            distance = hamming(value, candidate["dhash64"])
            if distance <= threshold:
                examples.append({
                    "train": candidate["image_path"],
                    "validation": row["image_path"],
                    "distance": distance,
                })
                break
        if len(examples) >= limit:
            break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-xml-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--windows-per-track", type=int, default=3)
    parser.add_argument("--minimum-track-agreement", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.window <= 5:
        raise ValueError("--window must be between 3 and 5")
    if not 0 <= args.near_duplicate_hamming <= 7:
        raise ValueError("--near-duplicate-hamming must be between 0 and 7")
    training_root = assert_training_only(args.training_root, "training root")
    xml_root = assert_training_only(args.training_xml_root, "training XML root")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
    source_archive_sha = sha256(args.source_archive)
    if source_archive_sha.lower() != args.expected_source_archive_sha256.lower():
        raise RuntimeError("source training archive SHA256 mismatch")
    evidence = {
        "source_archive": str(args.source_archive.resolve()),
        "source_archive_sha256": source_archive_sha,
        "source_index": str(args.source_index.resolve()),
        "source_index_sha256": sha256(args.source_index),
        "license_evidence": str(args.license_evidence.resolve()),
        "license_evidence_sha256": sha256(args.license_evidence),
    }

    observations, source_report = parse_training(training_root, xml_root)
    tracks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in observations:
        tracks[(row["sequence"], row["track_id"])].append(row)
    selected: list[dict] = []
    track_counts = Counter()
    source_type_counts = Counter()
    for (sequence, track_id), track in sorted(tracks.items()):
        source_type, agreement = source_type_consensus(track, args.minimum_track_agreement)
        windows = contiguous_windows(track, args.window, args.windows_per_track)
        if not windows:
            track_counts["rejected_no_contiguous_window"] += 1
            continue
        track_counts["accepted"] += 1
        source_type_counts[source_type] += 1
        for window_index, window in enumerate(windows):
            for position, row in enumerate(window):
                item = dict(row)
                item.update(
                    source_type_consensus=source_type,
                    source_type_agreement=agreement,
                    window_id=f"uadetrac-train:{sequence}:{track_id}:w{window_index}",
                    window_pos=position,
                )
                selected.append(item)

    grouped: dict[Path, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["image"]].append(row)
    output_root.mkdir(parents=True, exist_ok=True)

    def process_frame(source: Path, items: list[dict]) -> tuple[list[dict], Counter]:
        output: list[dict] = []
        rejected = Counter()
        frame = read_image(source)
        if frame is None:
            rejected["unreadable_frame"] += len(items)
            return output, rejected
        frame_height, frame_width = frame.shape[:2]
        for item in items:
            margin_x = item["box_width"] * args.margin
            margin_y = item["box_height"] * args.margin
            x1 = max(0, int(math.floor(item["left"] - margin_x)))
            y1 = max(0, int(math.floor(item["top"] - margin_y)))
            x2 = min(frame_width, int(math.ceil(item["left"] + item["box_width"] + margin_x)))
            y2 = min(frame_height, int(math.ceil(item["top"] + item["box_height"] + margin_y)))
            if x2 - x1 < 12 or y2 - y1 < 12:
                rejected["crop_too_small"] += 1
                continue
            crop = frame[y1:y2, x1:x2]
            sequence = item["sequence"]
            filename = f"{sequence}_t{item['track_id']:05d}_f{item['frame_number']:06d}.jpg"
            relative = Path("crops") / item["split"] / sequence / filename
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not write_jpeg(destination, crop):
                rejected["crop_write_failed"] += 1
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edge = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            crop_h, crop_w = crop.shape[:2]
            area = crop_h * crop_w
            vehicle_size = "small" if area < 128 * 96 else "medium" if area < 256 * 192 else "large"
            night = item["weather"] == "night"
            occluded = bool(item["occluded"])
            truncated = item["truncation_ratio"] > 0.0
            severe_visibility = occluded or item["truncation_ratio"] >= 0.35
            source_type = item["source_type_consensus"]
            exact_body = EXACT_BODY_MAP.get(source_type, "unknown")
            coarse_body = COARSE_BODY_MAP.get(source_type, "")
            if severe_visibility:
                exact_body = "unknown"
                coarse_body = ""
            body_supervised = exact_body != "unknown"
            tags: list[str] = []
            if night: tags.append("night")
            if item["weather"] == "rainy": tags.append("rain")
            if vehicle_size == "small": tags.append("small_target")
            if occluded: tags.append("occluded")
            if truncated: tags.append("truncated")
            if edge < 100.0: tags.append("low_detail")
            hard_score = min(8, 2 * len(tags))
            row = {
                "image_path": relative.as_posix(), "body_type": exact_body, "color": "unknown",
                "crop_quality": "usable" if severe_visibility or edge < 100.0 else "good",
                "viewpoint": "unknown", "blur": str(edge < 70.0).lower(),
                "occluded": str(occluded).lower(), "truncated": str(truncated).lower(),
                "night": str(night).lower(), "camera_id": f"uadetrac:{sequence}",
                "video_id": f"uadetrac:{sequence}",
                "track_group": f"uadetrac:{sequence}:{item['track_id']}", "split": item["split"],
                "source_frame_id": f"{sequence}:{item['frame_number']}", "review_status": "approved",
                "body_type_supervised": str(body_supervised).lower(), "color_supervised": "false",
                "annotation_source": "official_training_xml_track_consensus",
                "source_dataset": "UA-DETRAC", "source_manifest": str(item["source_xml"]),
                "source_license": "mirror-declared CC BY 4.0; upstream legal review required",
                "review_method": "official_track_id+source_type_consensus+visibility_fail_closed",
                "sha256": sha256(destination), "dhash64": dhash64(crop),
                "width": crop_w, "height": crop_h, "gray_mean": round(float(gray.mean()), 4),
                "gray_stddev": round(float(gray.std()), 4), "edge_variance": round(edge, 4),
                "occlusion_level": "occluded" if occluded else "truncated" if truncated else "visible",
                "vehicle_size": vehicle_size,
                "lighting": "night" if night else "low_light" if gray.mean() < 64 else "daylight",
                "label_confidence": "high" if (body_supervised or coarse_body) else "unknown",
                "license_train_eligible": "false", "source_group": sequence,
                "hard_example_priority": "high" if hard_score >= 4 else "medium" if hard_score else "normal",
                "hard_mining_tags": ";".join(tags), "hard_score": hard_score,
                "pseudo_label": "false", "pseudo_label_confidence": "",
                "frame_number": item["frame_number"], "window_id": item["window_id"],
                "window_pos": item["window_pos"], "source_vehicle_type": item["source_vehicle_type"],
                "source_color": item["source_color"], "weather": item["weather"],
                "truncation_ratio": item["truncation_ratio"], "track_body_truth": exact_body,
                "track_color_truth": "unknown", "track_label_agreement": round(item["source_type_agreement"], 6),
                "coarse_body_family": coarse_body, "source_partition": "training",
                "dedup_policy": f"sha256+dhash_hamming_le_{args.near_duplicate_hamming}",
                "_absolute_path": str(destination),
            }
            output.append(row)
        return output, rejected

    rows: list[dict] = []
    crop_rejections = Counter()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
        futures = [executor.submit(process_frame, source, items) for source, items in grouped.items()]
        for index, future in enumerate(as_completed(futures), 1):
            frame_rows, rejected = future.result()
            rows.extend(frame_rows)
            crop_rejections.update(rejected)
            if index % 2000 == 0:
                print(json.dumps({"processed_frames": index, "total_frames": len(futures), "crops": len(rows)}), flush=True)
    rows, dedup_report = deduplicate(rows, args.near_duplicate_hamming)
    rows.sort(key=lambda row: (row["split"], row["video_id"], row["track_group"], int(row["frame_number"])))
    post_filter_near_leaks = cross_split_near_duplicate_examples(
        rows, args.near_duplicate_hamming
    )

    manifest = output_root / "attribute_manifest.training-only.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    split_videos: dict[str, set[str]] = defaultdict(set)
    split_tracks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_videos[row["split"]].add(row["video_id"])
        split_tracks[row["split"]].add(row["track_group"])
    video_leaks = sorted(split_videos["train"] & split_videos["validation"])
    track_leaks = sorted(split_tracks["train"] & split_tracks["validation"])
    unexpected_source_colors = Counter(
        row["source_color"] for row in rows if row["source_color"] not in {"", "unknown"}
    )
    report = {
        "schema_version": "stage70-uadetrac-training-pool-v2",
        "created_at": utc_now(),
        "status": "pass" if rows and not video_leaks and not track_leaks and not post_filter_near_leaks else "fail",
        "eligibility": "research-only_non-deployable",
        "policy": {
            "source_partition": "training_only", "test_arguments_supported": False,
            "test_accessed": False, "frozen_video_used": False, "model_predictions_used": False,
            "exact_body_mapping": EXACT_BODY_MAP, "coarse_body_mapping": COARSE_BODY_MAP,
            "unmapped_type_policy": "unknown", "source_color_policy": "always unknown/not supervised",
            "visibility_policy": "occluded or truncation>=0.35 => exact and coarse body unknown",
            "minimum_track_agreement": args.minimum_track_agreement,
            "window_length": args.window, "maximum_windows_per_track": args.windows_per_track,
            "deduplication": f"global SHA256 and dHash Hamming<={args.near_duplicate_hamming}; source duplicates are removed before the post-filter leak gate",
            "license_train_eligible": False,
        },
        "evidence": evidence, "source_report": source_report,
        "observations_parsed": len(observations), "tracks_parsed": len(tracks),
        "track_counts": dict(sorted(track_counts.items())),
        "track_source_type_counts": dict(sorted(source_type_counts.items())),
        "selected_observations_before_crop": len(selected),
        "crop_rejections": dict(sorted(crop_rejections.items())),
        "deduplication": {
            **dedup_report,
            "post_filter_cross_split_examples": post_filter_near_leaks,
            "post_filter_cross_split_count": len(post_filter_near_leaks),
        },
        "written_rows": len(rows),
        "split_rows": dict(sorted(Counter(row["split"] for row in rows).items())),
        "split_videos": {key: len(value) for key, value in sorted(split_videos.items())},
        "split_tracks": {key: len(value) for key, value in sorted(split_tracks.items())},
        "body_counts": dict(sorted(Counter(row["body_type"] for row in rows if row["body_type_supervised"] == "true").items())),
        "coarse_body_counts": dict(sorted(Counter(row["coarse_body_family"] for row in rows if row["coarse_body_family"]).items())),
        "color_supervised_rows": sum(row["color_supervised"] == "true" for row in rows),
        "unexpected_source_colors": dict(sorted(unexpected_source_colors.items())),
        "weather_counts": dict(sorted(Counter(row["weather"] for row in rows).items())),
        "vehicle_size_counts": dict(sorted(Counter(row["vehicle_size"] for row in rows).items())),
        "hard_tag_counts": dict(sorted(Counter(tag for row in rows for tag in row["hard_mining_tags"].split(";") if tag).items())),
        "video_leaks": video_leaks, "track_leaks": track_leaks,
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "production_model_modified": False, "deployment_performed": False,
    }
    report_path = output_root / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
