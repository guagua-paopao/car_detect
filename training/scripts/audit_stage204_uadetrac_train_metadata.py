#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXACT_BODY_MAP = {"bus": "bus", "van": "van"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_weather(value: str) -> str:
    value = value.strip().lower()
    return value if value in {"sunny", "cloudy", "rainy", "night"} else "unknown"


def parse_sequence(xml_path: Path, sequence_dir: Path | None) -> tuple[dict, Counter, dict[tuple[str, str], Counter]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "sequence":
        raise RuntimeError(f"unexpected root tag in {xml_path}: {root.tag}")
    sequence = root.attrib.get("name", xml_path.stem)
    if sequence != xml_path.stem:
        raise RuntimeError(f"sequence name mismatch: {sequence} != {xml_path.stem}")
    sequence_attribute = root.find("sequence_attribute")
    weather = normalized_weather(sequence_attribute.attrib.get("sence_weather", "unknown") if sequence_attribute is not None else "unknown")
    frames = root.findall("frame")
    target_rows = 0
    class_rows = Counter()
    track_rows: dict[tuple[str, str], Counter] = defaultdict(Counter)
    malformed_targets = 0
    frame_numbers = set()
    for frame in frames:
        frame_number = frame.attrib.get("num", "")
        if not frame_number or frame_number in frame_numbers:
            raise RuntimeError(f"missing or duplicate frame number in {xml_path}: {frame_number}")
        frame_numbers.add(frame_number)
        target_list = frame.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            track_id = target.attrib.get("id", "")
            box = target.find("box")
            attribute = target.find("attribute")
            if not track_id or box is None or attribute is None:
                malformed_targets += 1
                continue
            try:
                left = float(box.attrib["left"])
                top = float(box.attrib["top"])
                width = float(box.attrib["width"])
                height = float(box.attrib["height"])
                valid_box = left >= 0 and top >= 0 and width > 0 and height > 0
            except (KeyError, ValueError):
                valid_box = False
            vehicle_type = attribute.attrib.get("vehicle_type", "unknown").strip().lower() or "unknown"
            if not valid_box:
                malformed_targets += 1
                continue
            target_rows += 1
            class_rows[vehicle_type] += 1
            track_rows[(sequence, track_id)][vehicle_type] += 1
    image_files = 0
    if sequence_dir is not None and sequence_dir.is_dir() and not sequence_dir.is_symlink():
        image_files = sum(
            1 for entry in os.scandir(sequence_dir)
            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    summary = {
        "sequence": sequence,
        "weather": weather,
        "xml_frames": len(frames),
        "image_directory": str(sequence_dir) if sequence_dir is not None else "",
        "image_files": image_files,
        "frame_count_matches": len(frames) == image_files,
        "target_rows": target_rows,
        "unique_tracks": len(track_rows),
        "malformed_targets": malformed_targets,
        "xml_bytes": xml_path.stat().st_size,
        "xml_sha256": sha256_path(xml_path),
    }
    return summary, class_rows, track_rows


def summarize_stage177(path: Path) -> dict:
    counts = Counter()
    classes = Counter()
    groups = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_dataset", "stage177_scene_label", "stage177_dedup_eligible", "stage177_effective_representative", "body_type", "stage177_group_key"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Stage177 manifest missing columns: {sorted(missing)}")
        for row in reader:
            if row["source_dataset"].strip().upper() != "UA-DETRAC":
                continue
            scene = row["stage177_scene_label"].strip().lower() or "unknown"
            counts["rows"] += 1
            counts[f"scene_{scene}"] += 1
            if row["stage177_dedup_eligible"].strip().lower() == "true":
                counts["dedup_eligible"] += 1
            if row["stage177_effective_representative"].strip().lower() == "true":
                counts["effective_representatives"] += 1
                counts[f"effective_scene_{scene}"] += 1
            classes[row["body_type"].strip().lower() or "unknown"] += 1
            groups.add(row["stage177_group_key"])
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_path(path),
        "counts": dict(sorted(counts.items())),
        "body_counts": dict(sorted(classes.items())),
        "distinct_groups": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--stage177-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    xml_root = (source_root / "original-xml" / "training").resolve()
    image_root = (source_root / "extracted" / "training").resolve()
    stage177 = args.stage177_manifest.resolve()
    output_root = args.output_root.resolve()
    for path in (source_root, xml_root, image_root, stage177, output_root):
        lowered = str(path).lower()
        if "vcas_rtsp_demo_60s" in lowered or "36-48" in lowered or "/test" in lowered or "\\test" in lowered:
            raise RuntimeError(f"prohibited path marker: {path}")
    if not xml_root.is_dir() or xml_root.is_symlink() or not image_root.is_dir() or image_root.is_symlink():
        raise RuntimeError("training metadata/image roots missing or symlinked")
    if not stage177.is_file() or stage177.is_symlink():
        raise RuntimeError("Stage177 manifest missing or symlinked")
    output_root.mkdir(parents=True, exist_ok=False)

    xml_paths = sorted(xml_root.glob("*.xml"))
    image_sequence_dirs = sorted(entry.name for entry in os.scandir(image_root) if entry.is_dir(follow_symlinks=False))
    if len(xml_paths) != 59 or len(image_sequence_dirs) != 60:
        raise RuntimeError(f"unexpected train inventory: xml={len(xml_paths)} image_sequences={len(image_sequence_dirs)}")

    image_dirs_by_sequence = {}
    for directory_name in image_sequence_dirs:
        match = re.search(r"(MVI_\d+)$", directory_name, flags=re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"cannot derive sequence id from image directory: {directory_name}")
        sequence_id = match.group(1).upper()
        if sequence_id in image_dirs_by_sequence:
            raise RuntimeError(f"duplicate image sequence id: {sequence_id}")
        image_dirs_by_sequence[sequence_id] = image_root / directory_name

    sequence_rows = []
    row_counts_by_weather = Counter()
    class_rows_by_weather: dict[str, Counter] = defaultdict(Counter)
    track_type_counts: dict[tuple[str, str], Counter] = {}
    track_weather: dict[tuple[str, str], str] = {}
    for xml_path in xml_paths:
        summary, class_rows, tracks = parse_sequence(xml_path, image_dirs_by_sequence.get(xml_path.stem.upper()))
        sequence_rows.append(summary)
        weather = summary["weather"]
        row_counts_by_weather[weather] += summary["target_rows"]
        class_rows_by_weather[weather].update(class_rows)
        for key, counts in tracks.items():
            track_type_counts[key] = counts
            track_weather[key] = weather

    xml_sequences = {row["sequence"] for row in sequence_rows}
    image_only_sequences = sorted(set(image_dirs_by_sequence) - {name.upper() for name in xml_sequences})
    xml_only_sequences = sorted({name.upper() for name in xml_sequences} - set(image_dirs_by_sequence))
    track_conflicts = 0
    track_counts_by_weather = Counter()
    track_class_counts_by_weather: dict[str, Counter] = defaultdict(Counter)
    cap5_rows_by_weather = Counter()
    cap5_exact_rows_by_weather = Counter()
    exact_tracks_by_weather = Counter()
    for key, counts in track_type_counts.items():
        weather = track_weather[key]
        track_counts_by_weather[weather] += 1
        if len(counts) != 1:
            track_conflicts += 1
            continue
        vehicle_type, row_count = next(iter(counts.items()))
        track_class_counts_by_weather[weather][vehicle_type] += 1
        cap5_rows_by_weather[weather] += min(row_count, 5)
        if vehicle_type in EXACT_BODY_MAP:
            exact_tracks_by_weather[weather] += 1
            cap5_exact_rows_by_weather[weather] += min(row_count, 5)

    sequence_manifest = output_root / "stage204-uadetrac-train-sequences.csv"
    with sequence_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)

    stage177_summary = summarize_stage177(stage177)
    report = {
        "schema_version": "stage204-uadetrac-train-metadata-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_train_metadata_inventory_no_pixels_opened",
        "scope": {
            "source_root": str(source_root),
            "train_xml_files": len(xml_paths),
            "train_image_sequence_directories": len(image_sequence_dirs),
            "image_only_sequences_without_xml": image_only_sequences,
            "xml_only_sequences_without_images": xml_only_sequences,
            "test_paths_opened": False,
            "train_pixels_opened": False,
        },
        "full_train_metadata": {
            "sequence_counts_by_weather": dict(sorted(Counter(row["weather"] for row in sequence_rows).items())),
            "xml_frame_counts_by_weather": dict(sorted(Counter({
                weather: sum(row["xml_frames"] for row in sequence_rows if row["weather"] == weather)
                for weather in {row["weather"] for row in sequence_rows}
            }).items())),
            "target_rows_by_weather": dict(sorted(row_counts_by_weather.items())),
            "class_rows_by_weather": {weather: dict(sorted(counts.items())) for weather, counts in sorted(class_rows_by_weather.items())},
            "unique_tracks_by_weather": dict(sorted(track_counts_by_weather.items())),
            "track_classes_by_weather": {weather: dict(sorted(counts.items())) for weather, counts in sorted(track_class_counts_by_weather.items())},
            "track_label_conflicts": track_conflicts,
            "track_cap5_effective_rows_by_weather": dict(sorted(cap5_rows_by_weather.items())),
            "exact_bus_van_tracks_by_weather": dict(sorted(exact_tracks_by_weather.items())),
            "exact_bus_van_track_cap5_rows_by_weather": dict(sorted(cap5_exact_rows_by_weather.items())),
        },
        "comparison_to_stage177_loader_manifest": stage177_summary,
        "outputs": {
            "sequence_manifest": str(sequence_manifest),
            "sequence_manifest_sha256": sha256_path(sequence_manifest),
        },
        "interpretation_policy": {
            "official_weather_night_is_sequence_level_natural_night_evidence": True,
            "car_is_coarse_passenger_vehicle_not_forced_to_sedan_suv_mpv": True,
            "others_is_unknown": True,
            "bus_and_van_are_provisional_exact_classes_pending_pixel_box_quality_teacher_and_cross_source_dedup": True,
            "training_authorized": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = output_root / "stage204-uadetrac-train-metadata-audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "SHA256SUMS").write_text(
        f"{sha256_path(report_path)}  {report_path.name}\n"
        f"{sha256_path(sequence_manifest)}  {sequence_manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
