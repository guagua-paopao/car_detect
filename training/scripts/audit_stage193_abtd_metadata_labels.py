#!/usr/bin/env python3
"""Audit ABTD v3 metadata and train labels without opening validation/test pixels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


EXPECTED = {
    "metadata.csv": (617698, "ceb9061436ee452d0888cde6d386f5e90a1fea4b28494eb97763a15a60322f2f"),
    "data.yaml": (195, "c24efd2d133d1b537629b2fb6c898b202eae3bd4d5494cd372679535d0f759b0"),
    "README_v3.txt": (4595, "0021ad7a4981c8186143737420aa0d48dbc7c52b22a51bff7a0f230d9b3afd9b"),
    "labels.zip": (2233642, "9f3c674ee6825ab72a9022a3395d834f6c32a655c2c6812dd2d5ef33ac21ce14"),
}
CLASSES = {0: "car", 1: "bus", 2: "truck", 3: "bike", 4: "rickshaw", 5: "van", 6: "bicycle", 7: "leguna", 8: "cng", 9: "emergency_vehicle"}
EXACT_BODY = {1: "bus", 2: "truck", 5: "van"}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> str:
    expected_size, expected_sha = EXPECTED[path.name]
    actual_sha = sha256(path)
    if path.stat().st_size != expected_size or actual_sha != expected_sha:
        raise RuntimeError(f"integrity mismatch: {path}")
    return actual_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {path.name: verify(path) for path in (args.metadata, args.labels, args.data_yaml, args.readme)}

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "split", "weather", "time_of_day", "illumination", "illumination_source", "augmented", "n_objects"}
    if len(rows) != 8605 or len({row["image"].casefold() for row in rows}) != 8605 or not required.issubset(rows[0]):
        raise RuntimeError("unexpected metadata contract")
    metadata = {Path(row["image"]).stem.casefold(): row for row in rows}
    split_counts = Counter(row["split"] for row in rows)
    original_train = [row for row in rows if row["split"] == "train" and row["augmented"] == "no"]
    scene_images = Counter((row["time_of_day"], row["illumination"], row["illumination_source"]) for row in original_train)

    box_counts: Counter[tuple[str, int]] = Counter()
    label_files = invalid_rows = missing_metadata = unsafe_members = 0
    with zipfile.ZipFile(args.labels) as archive:
        if archive.testzip():
            raise RuntimeError("labels ZIP CRC failure")
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                unsafe_members += 1
                continue
            if info.is_dir() or pure.suffix.lower() != ".txt":
                continue
            row = metadata.get(pure.stem.casefold())
            if row is None:
                missing_metadata += 1
                continue
            if row["split"] != "train" or row["augmented"] != "no":
                continue
            label_files += 1
            for line in archive.read(info).decode("utf-8").splitlines():
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    valid = len(parts) == 5 and class_id in CLASSES and all(0.0 <= value <= 1.0 for value in coords)
                except (ValueError, IndexError):
                    valid = False
                if not valid:
                    invalid_rows += 1
                    continue
                box_counts[("all", class_id)] += 1
                if row["time_of_day"] == "Night":
                    box_counts[("night", class_id)] += 1
                if row["illumination"] == "Low_Light":
                    box_counts[("lowlight", class_id)] += 1
                if row["time_of_day"] == "Night" or row["illumination"] == "Low_Light":
                    box_counts[("adverse_union", class_id)] += 1

    if split_counts != Counter({"train": 7183, "val": 949, "test": 473}) or len(original_train) != 3350:
        raise RuntimeError("published split counts mismatch")
    if label_files != 3350 or invalid_rows or missing_metadata or unsafe_members:
        raise RuntimeError("train label contract failed")

    def class_counts(scope: str) -> dict[str, int]:
        return {CLASSES[index]: box_counts[(scope, index)] for index in CLASSES}

    exact_night = sum(box_counts[("night", index)] for index in EXACT_BODY)
    exact_lowlight = sum(box_counts[("lowlight", index)] for index in EXACT_BODY)
    exact_adverse = sum(box_counts[("adverse_union", index)] for index in EXACT_BODY)
    report = {
        "schema_version": "stage193-abtd-metadata-label-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_metadata_labels_images_pending_no_training",
        "source": {
            "title": "Augmented Bangladesh Traffic Dataset for Deep Learning-based Vehicle Detection under Diverse Weather Conditions",
            "doi": "10.17632/2f9jp8bj45.3", "license": "CC BY 4.0", "version": 3,
            "capture": "nine real road intersections in Dhaka, Bangladesh, March 2025 to February 2026",
        },
        "inputs": {name: {"path": str(path), "sha256": hashes[path.name], "bytes": path.stat().st_size} for name, path in {
            "metadata": args.metadata, "labels": args.labels, "data_yaml": args.data_yaml, "readme": args.readme,
        }.items()},
        "metadata": {
            "rows": len(rows), "split_counts": dict(split_counts), "original_train_images": len(original_train),
            "original_train_scene_counts": {"|".join(key): value for key, value in sorted(scene_images.items())},
            "confirmed_natural_night_images": sum(value for (time, _, source), value in scene_images.items() if time == "Night" and source == "annotated"),
            "inferred_night_images": sum(value for (time, _, source), value in scene_images.items() if time == "Night" and source != "annotated"),
        },
        "train_labels": {
            "original_train_label_files": label_files, "invalid_rows": invalid_rows,
            "missing_metadata": missing_metadata, "unsafe_zip_members": unsafe_members,
            "all_boxes": class_counts("all"), "night_boxes": class_counts("night"),
            "lowlight_boxes": class_counts("lowlight"), "adverse_union_boxes": class_counts("adverse_union"),
            "exact_body_mapping": {CLASSES[key]: value for key, value in EXACT_BODY.items()},
            "exact_body_confirmed_natural_night_boxes": exact_night,
            "exact_body_lowlight_boxes": exact_lowlight,
            "exact_body_adverse_union_boxes": exact_adverse,
            "color_truth_boxes": 0,
        },
        "gates": {
            "metadata_and_label_integrity_pass": True, "train_originals_separated_from_augmentations": True,
            "validation_or_test_pixels_opened": False, "train_images_downloaded": False,
            "crop_quality_and_decontamination_complete": False, "training_authorized": False,
        },
        "policy": {
            "car_is_coarse_passenger_vehicle_not_sedan_or_suv": True,
            "rickshaw_bike_bicycle_leguna_cng_emergency_vehicle_not_mapped_to_exact_project_body": True,
            "night_requires_annotated_time_of_day": True, "inferred_lowlight_kept_separate": True,
            "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
        "next_action": "download the 1.89GB train image ZIP only, verify SHA/CRC, open original train images only, crop exact bus/truck/van boxes, group by source filename sequence, and decontaminate against existing train data",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "night_images": report["metadata"]["confirmed_natural_night_images"], "exact_night_boxes": exact_night, "exact_adverse_boxes": exact_adverse}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
