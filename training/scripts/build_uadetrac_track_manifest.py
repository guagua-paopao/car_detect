#!/usr/bin/env python3
"""Build conservative same-vehicle attribute windows from UA-DETRAC XML."""

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
COLOR_MAP = {
    "black": "black",
    "white": "white",
    "gray": "silver_gray",
    "grey": "silver_gray",
    "silver": "silver_gray",
    "blue": "blue",
    "red": "red",
    "brown": "brown_beige",
    "green": "green",
    "yellow": "yellow_orange",
    "orange": "yellow_orange",
    "others": "other",
    "other": "other",
}
BODY_MAP = {"bus": "bus", "van": "van"}


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
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
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


def stable_split(sequence: str, partition: str) -> str:
    if partition == "test":
        return "test"
    bucket = int(hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:8], 16) % 5
    return "validation" if bucket == 0 else "train"


def mapped_consensus(values: list[str], mapping: dict[str, str], minimum_share: float) -> tuple[str, float]:
    mapped = [mapping.get(value.strip().lower(), "unknown") for value in values]
    known = [value for value in mapped if value != "unknown"]
    if not known:
        return "unknown", 0.0
    label, count = Counter(known).most_common(1)[0]
    share = count / len(mapped)
    return (label, share) if share >= minimum_share else ("unknown", share)


def contiguous_windows(observations: list[dict], length: int, maximum: int) -> list[list[dict]]:
    observations = sorted(observations, key=lambda row: row["frame_number"])
    candidates = []
    for index in range(0, len(observations) - length + 1):
        window = observations[index:index + length]
        frames = [row["frame_number"] for row in window]
        if all(b - a == 1 for a, b in zip(frames, frames[1:])):
            candidates.append(window)
    if len(candidates) <= maximum:
        proposed = candidates
    else:
        indexes = []
        for point in np.linspace(0, len(candidates) - 1, maximum):
            index = int(round(float(point)))
            if index not in indexes:
                indexes.append(index)
        proposed = [candidates[index] for index in indexes]
    accepted: list[list[dict]] = []
    used_frames: set[int] = set()
    for window in proposed:
        frames = {row["frame_number"] for row in window}
        if frames & used_frames:
            continue
        accepted.append(window)
        used_frames.update(frames)
    return accepted


def parse_partition(image_root: Path, xml_root_path: Path, partition: str) -> tuple[list[dict], dict]:
    xml_paths = sorted(xml_root_path.rglob("*.xml"))
    image_dirs: dict[str, list[Path]] = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_dir():
            name = sequence_name(path.name)
            if name:
                image_dirs[name].append(path)
    rows: list[dict] = []
    report = {
        "partition": partition,
        "image_root": str(image_root),
        "xml_root": str(xml_root_path),
        "xml_files": len(xml_paths),
        "missing_image_sequences": [],
    }
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
        sequence = sequence_name(xml_root.attrib.get("name", "") or xml_path.stem)
        if not sequence:
            continue
        candidate_dirs = image_dirs.get(sequence, [])
        image_index: dict[int, Path] = {}
        for directory in candidate_dirs:
            for image in directory.iterdir():
                if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    number = frame_number(image)
                    if number is not None:
                        image_index.setdefault(number, image)
        if not image_index:
            report["missing_image_sequences"].append(sequence)
            continue
        sequence_attribute = xml_root.find("sequence_attribute")
        weather = "unknown"
        camera_state = "unknown"
        if sequence_attribute is not None:
            weather = str(
                sequence_attribute.attrib.get(
                    "weather", sequence_attribute.attrib.get("sence_weather", "unknown")
                )
            ).strip().lower()
            camera_state = str(sequence_attribute.attrib.get("camera_state", "unknown")).strip().lower()
        for frame in xml_root.findall("frame"):
            try:
                number = int(frame.attrib["num"])
            except (KeyError, ValueError):
                continue
            image = image_index.get(number)
            if image is None:
                continue
            target_list = frame.find("target_list")
            if target_list is None:
                continue
            for target in target_list.findall("target"):
                box = target.find("box")
                attribute = target.find("attribute")
                if box is None or attribute is None:
                    continue
                try:
                    left = float(box.attrib["left"]); top = float(box.attrib["top"])
                    width = float(box.attrib["width"]); height = float(box.attrib["height"])
                    track_id = int(target.attrib["id"])
                    truncation = float(attribute.attrib.get("truncation_ratio", 0.0))
                except (KeyError, ValueError):
                    continue
                occlusion = target.find("occlusion")
                occluded = occlusion is not None and len(list(occlusion)) > 0
                rows.append({
                    "partition": partition,
                    "sequence": sequence,
                    "split": stable_split(sequence, partition),
                    "camera_state": camera_state,
                    "weather": weather,
                    "frame_number": number,
                    "image": image,
                    "track_id": track_id,
                    "left": left,
                    "top": top,
                    "box_width": width,
                    "box_height": height,
                    "source_vehicle_type": str(attribute.attrib.get("vehicle_type", "unknown")).strip().lower(),
                    "source_color": str(attribute.attrib.get("color", "unknown")).strip().lower(),
                    "truncation_ratio": truncation,
                    "occluded": occluded,
                    "source_xml": xml_path,
                })
    report["observations"] = len(rows)
    report["sequences_with_images"] = len({row["sequence"] for row in rows})
    return rows, report


def dhash64(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_jpeg(path: Path, image: np.ndarray, quality: int = 95) -> bool:
    try:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--training-xml-root", type=Path)
    parser.add_argument("--test-xml-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--windows-per-track", type=int, default=2)
    parser.add_argument("--minimum-track-agreement", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.window < 3 or args.window > 5:
        raise ValueError("window must be between 3 and 5")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {args.output_root}")

    training_rows, training_report = parse_partition(
        args.training_root, args.training_xml_root or args.training_root, "training"
    )
    test_rows, test_report = parse_partition(
        args.test_root, args.test_xml_root or args.test_root, "test"
    )
    observations = training_rows + test_rows
    tracks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in observations:
        tracks[(row["sequence"], row["track_id"])].append(row)

    selected: list[dict] = []
    track_report = Counter()
    color_truth_counts = Counter()
    body_truth_counts = Counter()
    for (sequence, track_id), track in sorted(tracks.items()):
        color_truth, color_share = mapped_consensus(
            [row["source_color"] for row in track], COLOR_MAP, args.minimum_track_agreement
        )
        body_truth, body_share = mapped_consensus(
            [row["source_vehicle_type"] for row in track], BODY_MAP, args.minimum_track_agreement
        )
        windows = contiguous_windows(track, args.window, args.windows_per_track)
        if not windows:
            track_report["rejected_no_contiguous_window"] += 1
            continue
        track_report["accepted_tracks"] += 1
        color_truth_counts[color_truth] += 1
        body_truth_counts[body_truth] += 1
        for window_index, window in enumerate(windows):
            for position, row in enumerate(window):
                item = dict(row)
                item.update(
                    track_color_truth=color_truth,
                    track_body_truth=body_truth,
                    color_agreement=color_share,
                    body_agreement=body_share,
                    window_id=f"uadetrac:{sequence}:{track_id}:w{window_index}",
                    window_pos=position,
                )
                selected.append(item)

    grouped: dict[Path, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["image"]].append(row)
    args.output_root.mkdir(parents=True, exist_ok=True)

    def process_frame(source: Path, items: list[dict]) -> tuple[list[dict], Counter]:
        output_rows: list[dict] = []
        rejected = Counter()
        frame = read_image(source)
        if frame is None:
            rejected["unreadable_frame"] += len(items)
            return output_rows, rejected
        frame_height, frame_width = frame.shape[:2]
        for item in items:
            left = item["left"]; top = item["top"]
            width = item["box_width"]; height = item["box_height"]
            margin_x = width * args.margin; margin_y = height * args.margin
            x1 = max(0, int(math.floor(left - margin_x))); y1 = max(0, int(math.floor(top - margin_y)))
            x2 = min(frame_width, int(math.ceil(left + width + margin_x)))
            y2 = min(frame_height, int(math.ceil(top + height + margin_y)))
            if x2 - x1 < 12 or y2 - y1 < 12:
                rejected["crop_too_small"] += 1
                continue
            crop = frame[y1:y2, x1:x2]
            split = item["split"]
            sequence = item["sequence"]
            filename = f"{sequence}_t{item['track_id']:05d}_f{item['frame_number']:06d}.jpg"
            relative = Path("crops") / split / sequence / filename
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not write_jpeg(destination, crop, 95):
                rejected["crop_write_failed"] += 1
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edge = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            crop_h, crop_w = crop.shape[:2]
            area = crop_w * crop_h
            vehicle_size = "small" if area < 128 * 96 else "medium" if area < 256 * 192 else "large"
            night = item["weather"] == "night"
            occluded = bool(item["occluded"])
            truncated = item["truncation_ratio"] > 0.0
            hard_tags = []
            if night: hard_tags.append("night")
            if vehicle_size == "small": hard_tags.append("small_target")
            if occluded: hard_tags.append("occluded")
            if truncated: hard_tags.append("truncated")
            if edge < 100.0: hard_tags.append("low_detail")
            hard_score = min(8, 2 * len(hard_tags))
            severe_visibility = occluded or item["truncation_ratio"] >= 0.35
            color_supervised = item["track_color_truth"] != "unknown" and not severe_visibility
            body_supervised = item["track_body_truth"] != "unknown" and not severe_visibility
            output_rows.append({
                "image_path": relative.as_posix(),
                "body_type": item["track_body_truth"] if body_supervised else "unknown",
                "color": item["track_color_truth"] if color_supervised else "unknown",
                "crop_quality": "usable" if severe_visibility or edge < 100.0 else "good",
                "viewpoint": "unknown",
                "blur": str(edge < 70.0).lower(),
                "occluded": str(occluded).lower(),
                "truncated": str(truncated).lower(),
                "night": str(night).lower(),
                "camera_id": f"uadetrac:{sequence}",
                "video_id": f"uadetrac:{sequence}",
                "track_group": f"uadetrac:{sequence}:{item['track_id']}",
                "split": split,
                "source_frame_id": f"{sequence}:{item['frame_number']}",
                "review_status": "approved",
                "body_type_supervised": str(body_supervised).lower(),
                "color_supervised": str(color_supervised).lower(),
                "annotation_source": "official_xml_track_consensus",
                "source_dataset": "UA-DETRAC",
                "source_manifest": str(item["source_xml"]),
                "source_license": "mirror-declared CC BY 4.0; upstream legal review required",
                "review_method": "official_track_id+>=0.8_label_consensus+visibility_fail_closed",
                "sha256": file_sha256(destination),
                "dhash64": dhash64(crop),
                "width": crop_w,
                "height": crop_h,
                "gray_mean": round(float(gray.mean()), 4),
                "gray_stddev": round(float(gray.std()), 4),
                "edge_variance": round(edge, 4),
                "occlusion_level": "occluded" if occluded else "truncated" if truncated else "visible",
                "vehicle_size": vehicle_size,
                "lighting": "night" if night else "low_light" if gray.mean() < 64 else "daylight",
                "label_confidence": "high" if not severe_visibility else "unknown",
                "license_train_eligible": "false",
                "source_group": sequence,
                "hard_example_priority": "high" if hard_score >= 4 else "medium" if hard_score else "normal",
                "hard_mining_tags": ";".join(hard_tags),
                "hard_score": hard_score,
                "pseudo_label": "false",
                "pseudo_label_confidence": "",
                "frame_number": item["frame_number"],
                "window_id": item["window_id"],
                "window_pos": item["window_pos"],
                "source_vehicle_type": item["source_vehicle_type"],
                "source_color": item["source_color"],
                "weather": item["weather"],
                "truncation_ratio": item["truncation_ratio"],
                "track_body_truth": item["track_body_truth"],
                "track_color_truth": item["track_color_truth"],
                "track_label_agreement": round(min(item["body_agreement"], item["color_agreement"]), 6),
            })
        return output_rows, rejected

    rows: list[dict] = []
    rejected = Counter()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
        futures = {executor.submit(process_frame, source, items): source for source, items in grouped.items()}
        for index, future in enumerate(as_completed(futures), 1):
            frame_rows, frame_rejected = future.result()
            rows.extend(frame_rows); rejected.update(frame_rejected)
            if index % 2000 == 0:
                print(json.dumps({"processed_frames": index, "total_frames": len(futures), "crops": len(rows)}), flush=True)
    rows.sort(key=lambda row: (row["split"], row["video_id"], row["track_group"], int(row["frame_number"])))

    manifest = args.output_root / "attribute_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    split_videos = defaultdict(set); split_tracks = defaultdict(set)
    for row in rows:
        split_videos[row["split"]].add(row["video_id"])
        split_tracks[row["split"]].add(row["track_group"])
    split_names = sorted(split_videos)
    video_leaks = []
    track_leaks = []
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            video_leaks.extend(sorted(split_videos[left] & split_videos[right]))
            track_leaks.extend(sorted(split_tracks[left] & split_tracks[right]))
    report = {
        "schema_version": "uadetrac-attribute-tracks-v1",
        "created_at": now(),
        "status": "pass" if rows and not video_leaks and not track_leaks else "fail",
        "policy": {
            "frozen_video_used": False,
            "model_predictions_used": False,
            "official_track_ids_used": True,
            "window_length": args.window,
            "maximum_windows_per_track": args.windows_per_track,
            "minimum_track_label_agreement": args.minimum_track_agreement,
            "body_mapping": BODY_MAP,
            "unmapped_car_policy": "unknown/body_type_supervised=false",
            "visibility_policy": "occluded or truncation>=0.35 => attributes unknown for that frame",
            "license_train_eligible": False,
            "color_ground_truth_available": any(row["source_color"] != "unknown" for row in observations),
        },
        "source_reports": [training_report, test_report],
        "observations_parsed": len(observations),
        "tracks_parsed": len(tracks),
        "track_filter_counts": dict(sorted(track_report.items())),
        "track_body_truth_counts": dict(sorted(body_truth_counts.items())),
        "track_color_truth_counts": dict(sorted(color_truth_counts.items())),
        "selected_observations": len(selected),
        "written_rows": len(rows),
        "rejected_crop_counts": dict(sorted(rejected.items())),
        "split_rows": dict(sorted(Counter(row["split"] for row in rows).items())),
        "split_videos": {key: len(value) for key, value in sorted(split_videos.items())},
        "split_tracks": {key: len(value) for key, value in sorted(split_tracks.items())},
        "weather_counts": dict(sorted(Counter(row["weather"] for row in rows).items())),
        "color_counts": dict(sorted(Counter(row["color"] for row in rows if row["color_supervised"] == "true").items())),
        "body_counts": dict(sorted(Counter(row["body_type"] for row in rows if row["body_type_supervised"] == "true").items())),
        "hard_tag_counts": dict(sorted(Counter(tag for row in rows for tag in row["hard_mining_tags"].split(";") if tag).items())),
        "video_leaks": video_leaks,
        "track_leaks": track_leaks,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "source_license": "mirror-declared CC BY 4.0; upstream legal review required before production training use",
    }
    report_path = args.output_root / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
