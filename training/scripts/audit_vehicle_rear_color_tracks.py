#!/usr/bin/env python3
"""Audit Vehicle-Rear color/track truth without exposing plate identifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_archive_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def normalize_color(value) -> str:
    color = str(value or "unknown").strip().lower()
    return color or "unknown"


def audit_pair_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Vehicle-Rear pair JSON must be an object keyed by set")
    pair_labels: Counter[str] = Counter()
    metadata_colors: Counter[str] = Counter()
    image_color: dict[str, str] = {}
    conflicts: Counter[str] = Counter()
    track_images: dict[str, set[str]] = defaultdict(set)
    cameras: Counter[str] = Counter()
    sets: Counter[str] = Counter()
    entries = 0
    invalid_entries = 0
    unsafe_paths = 0
    for set_name, pairs in data.items():
        if not isinstance(pairs, list):
            invalid_entries += 1
            continue
        sets[str(set_name)] += len(pairs)
        for entry in pairs:
            entries += 1
            if not isinstance(entry, list) or len(entry) < 7:
                invalid_entries += 1
                continue
            pair_labels[str(entry[4])] += 1
            for image_list_index, metadata_index in ((1, 5), (3, 6)):
                images = entry[image_list_index]
                metadata = entry[metadata_index]
                if not isinstance(images, list) or not isinstance(metadata, dict):
                    invalid_entries += 1
                    continue
                color = normalize_color(metadata.get("color"))
                metadata_colors[color] += 1
                for raw_path in images:
                    archive_path = str(raw_path)
                    if not safe_archive_path(archive_path):
                        unsafe_paths += 1
                        continue
                    parts = PurePosixPath(archive_path).parts
                    if "classes_carros" not in parts:
                        continue
                    previous = image_color.setdefault(archive_path, color)
                    if previous != color:
                        conflicts[f"{previous}->{color}"] += 1
                    identity_path = str(PurePosixPath(archive_path).parent)
                    track_images[identity_path].add(archive_path)
                    for part in parts:
                        if part.startswith("Camera"):
                            cameras[part] += 1
    unique_color_counts = Counter(image_color.values())
    track_color_counts: Counter[str] = Counter()
    for identity, images in track_images.items():
        colors = Counter(image_color[path] for path in images)
        track_color_counts[colors.most_common(1)[0][0]] += 1
    return {
        "entries": entries,
        "invalid_entries": invalid_entries,
        "pair_label_counts": dict(sorted(pair_labels.items())),
        "metadata_color_occurrences": dict(sorted(metadata_colors.items())),
        "unique_vehicle_image_paths": len(image_color),
        "unique_vehicle_image_color_counts": dict(sorted(unique_color_counts.items())),
        "unique_track_groups": len(track_images),
        "track_color_counts": dict(sorted(track_color_counts.items())),
        "track_length_min": min(map(len, track_images.values()), default=0),
        "track_length_max": max(map(len, track_images.values()), default=0),
        "track_length_mean": (
            sum(map(len, track_images.values())) / len(track_images) if track_images else 0.0
        ),
        "color_conflicts": dict(sorted(conflicts.items())),
        "unsafe_paths": unsafe_paths,
        "camera_path_occurrences": dict(sorted(cameras.items())),
        "set_pair_counts": dict(sorted(sets.items())),
    }


def audit_xml(root: Path) -> dict:
    colors: Counter[str] = Counter()
    usable_colors: Counter[str] = Counter()
    motorcycle = Counter()
    quality = Counter()
    discard = Counter()
    files = 0
    vehicles = 0
    for path in sorted(root.rglob("*.xml")):
        files += 1
        tree = ET.parse(path)
        for vehicle in tree.findall(".//vehicle"):
            vehicles += 1
            color = normalize_color(vehicle.attrib.get("color"))
            colors[color] += 1
            is_moto = vehicle.attrib.get("moto", "False").strip().lower() == "true"
            is_quality = vehicle.attrib.get("quality", "False").strip().lower() == "true"
            is_discard = vehicle.attrib.get("discard", "False").strip().lower() == "true"
            motorcycle[str(is_moto).lower()] += 1
            quality[str(is_quality).lower()] += 1
            discard[str(is_discard).lower()] += 1
            if not is_moto and is_quality and not is_discard:
                usable_colors[color] += 1
    return {
        "xml_files": files,
        "vehicles": vehicles,
        "color_counts": dict(sorted(colors.items())),
        "usable_non_motorcycle_color_counts": dict(sorted(usable_colors.items())),
        "motorcycle_counts": dict(sorted(motorcycle.items())),
        "quality_counts": dict(sorted(quality.items())),
        "discard_counts": dict(sorted(discard.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--xml-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = audit_pair_json(args.pair_json)
    xml = audit_xml(args.xml_root)
    status = "pass_quarantine_label_audit"
    if pair["invalid_entries"] or pair["unsafe_paths"] or pair["color_conflicts"]:
        status = "fail_label_or_path_integrity"
    report = {
        "schema_version": "vehicle-rear-color-track-audit-v1",
        "status": status,
        "pair_json": str(args.pair_json.resolve()),
        "pair_json_sha256": sha256(args.pair_json),
        "pair_audit": pair,
        "xml_root": str(args.xml_root.resolve()),
        "xml_audit": xml,
        "privacy_policy": "raw plate identifiers are never emitted in this report; only aggregate track counts are recorded",
        "training_policy": {
            "allowed": False,
            "reason": "dataset-specific license applicability, image extraction, plate masking, deduplication and grouped split construction remain pending",
            "frozen_video_used": False,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
