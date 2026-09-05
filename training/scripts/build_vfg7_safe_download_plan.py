#!/usr/bin/env python3
"""Build a fail-closed VFG-7 image download and crop evaluation plan.

VFG-7 is CC BY-NC 4.0, so every emitted row is evaluation-only.  Structured
attributes are bound to YOLO boxes only when the per-image coarse-class group
is unambiguous: singleton groups bind directly, while multi-box groups may
share an attribute only if every annotation has agreement 1.0 and the mapped
attribute is identical.  Track IDs are emitted only for singleton groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_vfg7_vlm_alignment import body_label, mapped_color, type_class


REVISION = "bf7e035e0cc9660b31c2c01301d31aed8916ec2a"
REPO_ID = "Telody1220/VFG-7"
LICENSE = "CC BY-NC 4.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(parts[0])
        coords = tuple(float(value) for value in parts[1:])
        if class_id < 0 or class_id > 6 or any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{path}:{line_number}: invalid YOLO row")
        rows.append((class_id, *coords))
    return rows


def read_lfs_pointer(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="ascii")
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if not oid or not size:
        raise ValueError(f"not a valid Git LFS pointer: {path}")
    return oid.group(1), int(size.group(1))


def source_video(stem: str) -> str:
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.metadata_root.resolve()
    output_dir = args.output_dir.resolve()
    image_output_root = args.image_output_root.resolve()
    vlm_path = root / "vlm_annotations.json"
    annotations = json.loads(vlm_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in annotations:
        grouped[str(item.get("image", ""))].append(item)

    labels: dict[str, tuple[str, Path]] = {}
    duplicate_stems = []
    for split in ("train", "val"):
        for path in sorted((root / "labels" / split).glob("*.txt")):
            if path.stem in labels:
                duplicate_stems.append(path.stem)
            labels[path.stem] = (split, path)
    if duplicate_stems:
        raise RuntimeError(f"duplicate image stems across splits: {duplicate_stems[:10]}")

    counters = Counter()
    rows: list[dict] = []
    rejected_examples = []
    for image_name, items in sorted(grouped.items()):
        stem = Path(image_name).stem
        label_entry = labels.get(stem)
        if not label_entry:
            counters["missing_label_images"] += 1
            continue
        split, label_path = label_entry
        image_rel = Path("images") / split / image_name
        pointer_path = root / image_rel
        if not pointer_path.is_file():
            counters["missing_lfs_pointer_images"] += 1
            continue
        image_sha, image_size = read_lfs_pointer(pointer_path)
        yolo = read_yolo(label_path)
        yolo_by_class: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(yolo):
            yolo_by_class[row[0]].append(index)
        items_by_class: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            class_id = type_class(str(item.get("vehicle_type", "")))
            if class_id is not None:
                items_by_class[class_id].append(item)

        for class_id in sorted(set(yolo_by_class) | set(items_by_class)):
            box_indexes = yolo_by_class.get(class_id, [])
            candidates = items_by_class.get(class_id, [])
            if not box_indexes or len(box_indexes) != len(candidates):
                counters["rejected_group_count_mismatch_boxes"] += max(len(box_indexes), len(candidates))
                if len(rejected_examples) < 20:
                    rejected_examples.append({
                        "image": image_name, "class_id": class_id,
                        "box_count": len(box_indexes), "annotation_count": len(candidates),
                    })
                continue
            agreements = [float(item.get("agreement", 0.0) or 0.0) for item in candidates]
            body_values = [body_label(str(item.get("vehicle_type", ""))) for item in candidates]
            color_values = [mapped_color(str(item.get("color", ""))) for item in candidates]
            body_safe = min(agreements) >= 0.999 and len(set(body_values)) == 1 and body_values[0] != "unknown"
            color_safe = min(agreements) >= 0.999 and len(set(color_values)) == 1 and color_values[0] != "unknown"
            if not (body_safe or color_safe):
                counters["rejected_ambiguous_attribute_boxes"] += len(box_indexes)
                continue
            method = "singleton_unique" if len(box_indexes) == 1 else "homogeneous_class_group"
            track_id = str(candidates[0].get("track_id", "")) if len(box_indexes) == 1 else ""
            video = source_video(stem)
            for box_index in box_indexes:
                _, x_center, y_center, width, height = yolo[box_index]
                rows.append({
                    "dataset": "VFG-7",
                    "repo_id": REPO_ID,
                    "revision": REVISION,
                    "license": LICENSE,
                    "training_eligible": "false",
                    "usage": "independent_evaluation_only",
                    "source_split": split,
                    "evaluation_split": "validation" if split == "train" else "test",
                    "source_video": video,
                    "image_name": image_name,
                    "image_repo_path": image_rel.as_posix(),
                    "image_sha256": image_sha,
                    "image_size": image_size,
                    "image_local_path": str(image_output_root / split / image_name),
                    "label_repo_path": label_path.relative_to(root).as_posix(),
                    "yolo_row_index": box_index,
                    "yolo_class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "body_label": body_values[0] if body_safe else "unknown",
                    "color_label": color_values[0] if color_safe else "unknown",
                    "body_supervised": str(body_safe).lower(),
                    "color_supervised": str(color_safe).lower(),
                    "track_id": track_id,
                    "track_key": f"{video}:{track_id}" if track_id else "",
                    "agreement_min": min(agreements),
                    "alignment_method": method,
                })

    validation_videos = {row["source_video"] for row in rows if row["evaluation_split"] == "validation"}
    test_videos = {row["source_video"] for row in rows if row["evaluation_split"] == "test"}
    validation_tracks = {row["track_key"] for row in rows if row["evaluation_split"] == "validation" and row["track_key"]}
    test_tracks = {row["track_key"] for row in rows if row["evaluation_split"] == "test" and row["track_key"]}
    validation_raw_tracks = {row["track_id"] for row in rows if row["evaluation_split"] == "validation" and row["track_id"]}
    test_raw_tracks = {row["track_id"] for row in rows if row["evaluation_split"] == "test" and row["track_id"]}
    overlapping_videos = validation_videos & test_videos
    overlapping_tracks = validation_tracks & test_tracks
    overlapping_raw_tracks = validation_raw_tracks & test_raw_tracks

    quarantined = []
    safe_rows = []
    for row in rows:
        reasons = []
        if row["evaluation_split"] == "validation" and row["source_video"] in overlapping_videos:
            reasons.append("source_video_overlaps_test")
        if row["evaluation_split"] == "validation" and row["track_key"] in overlapping_tracks:
            reasons.append("track_key_overlaps_test")
        if row["evaluation_split"] == "validation" and row["track_id"] in overlapping_raw_tracks:
            reasons.append("raw_track_id_overlaps_test")
        if reasons:
            quarantined.append({**row, "quarantine_reason": ";".join(reasons)})
        else:
            safe_rows.append(row)

    # Re-audit leakage after preserving test and quarantining any conflicting validation groups.
    final_validation_videos = {row["source_video"] for row in safe_rows if row["evaluation_split"] == "validation"}
    final_test_videos = {row["source_video"] for row in safe_rows if row["evaluation_split"] == "test"}
    final_validation_tracks = {row["track_key"] for row in safe_rows if row["evaluation_split"] == "validation" and row["track_key"]}
    final_test_tracks = {row["track_key"] for row in safe_rows if row["evaluation_split"] == "test" and row["track_key"]}
    if final_validation_videos & final_test_videos or final_validation_tracks & final_test_tracks:
        raise RuntimeError("leakage remains after quarantine")

    image_rows: dict[str, dict] = {}
    for row in safe_rows:
        key = row["image_repo_path"]
        existing = image_rows.get(key)
        item = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "repo_path": key,
            "sha256": row["image_sha256"],
            "size": row["image_size"],
            "destination": row["image_local_path"],
        }
        if existing and existing != item:
            raise RuntimeError(f"conflicting image metadata: {key}")
        image_rows[key] = item

    body_counts = Counter(row["body_label"] for row in safe_rows if row["body_supervised"] == "true")
    color_counts = Counter(row["color_label"] for row in safe_rows if row["color_supervised"] == "true")
    split_counts = Counter(row["evaluation_split"] for row in safe_rows)
    track_colors: dict[str, list[str]] = defaultdict(list)
    for row in safe_rows:
        if row["track_key"] and row["color_supervised"] == "true":
            track_colors[row["track_key"]].append(row["color_label"])
    safe_color_tracks = sum(len(values) >= 3 and len(set(values)) == 1 for values in track_colors.values())
    expected_bytes = sum(item["size"] for item in image_rows.values())

    fields = [
        "dataset", "repo_id", "revision", "license", "training_eligible", "usage",
        "source_split", "evaluation_split", "source_video", "image_name", "image_repo_path",
        "image_sha256", "image_size", "image_local_path", "label_repo_path", "yolo_row_index",
        "yolo_class_id", "x_center", "y_center", "width", "height", "body_label", "color_label",
        "body_supervised", "color_supervised", "track_id", "track_key", "agreement_min",
        "alignment_method",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_plan_path = output_dir / "vfg7-safe-crop-plan.csv"
    quarantine_path = output_dir / "vfg7-quarantined-validation-rows.csv"
    download_path = output_dir / "vfg7-safe-download-plan.jsonl"
    report_path = output_dir / "vfg7-safe-plan-report.json"
    write_csv(crop_plan_path, safe_rows, fields)
    write_csv(quarantine_path, quarantined, fields + ["quarantine_reason"])
    with download_path.open("w", encoding="utf-8") as handle:
        for item in sorted(image_rows.values(), key=lambda value: value["repo_path"]):
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    status = "pass_partial_fail_closed" if sum(color_counts.values()) >= 1000 and safe_color_tracks >= 100 else "fail_closed"
    report = {
        "schema_version": "vfg7-safe-download-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": {"repo_id": REPO_ID, "revision": REVISION, "license": LICENSE},
        "policy": {
            "training_eligible": False,
            "usage": "independent_evaluation_only",
            "original_train_usage": "validation_threshold_selection_only",
            "original_val_usage": "withheld_test_only",
            "test_not_used_for_thresholds": True,
            "frozen_video_used": False,
            "model_predictions_used": False,
            "attribute_minimum_agreement": 1.0,
            "track_id_policy": "only singleton coarse-class groups retain track IDs",
            "leakage_policy": "preserve original val as test; quarantine conflicting original-train video/track groups",
        },
        "input": {
            "metadata_root": str(root),
            "vlm_annotations": len(annotations),
            "vlm_annotations_sha256": sha256(vlm_path),
            "label_files": len(labels),
            "lfs_pointer_files": sum(1 for _ in (root / "images").rglob("*.jpg")),
        },
        "output": {
            "crop_rows": len(safe_rows),
            "crop_rows_by_evaluation_split": dict(sorted(split_counts.items())),
            "unique_images": len(image_rows),
            "expected_download_bytes": expected_bytes,
            "expected_download_gib": expected_bytes / (1024 ** 3),
            "body_supervised_rows": sum(body_counts.values()),
            "body_counts": dict(sorted(body_counts.items())),
            "color_supervised_rows": sum(color_counts.values()),
            "color_counts": dict(sorted(color_counts.items())),
            "safe_consistent_color_tracks_with_at_least_3_observations": safe_color_tracks,
            "quarantined_validation_rows": len(quarantined),
        },
        "leakage_audit": {
            "pre_quarantine_overlapping_videos": sorted(overlapping_videos),
            "pre_quarantine_overlapping_track_keys": sorted(overlapping_tracks),
            "pre_quarantine_overlapping_raw_track_ids": sorted(overlapping_raw_tracks),
            "post_quarantine_video_overlap_count": len(final_validation_videos & final_test_videos),
            "post_quarantine_track_overlap_count": len(final_validation_tracks & final_test_tracks),
        },
        "rejection_counters": dict(sorted(counters.items())),
        "rejection_examples": rejected_examples,
        "artifacts": {
            "crop_plan": str(crop_plan_path),
            "quarantine_plan": str(quarantine_path),
            "download_plan": str(download_path),
        },
        "decision": (
            "download only the checksum-pinned safe image subset; build evaluation crops; never add rows to production training"
            if status == "pass_partial_fail_closed" else
            "do not download: safe color or repeated-track evidence fell below the fail-closed minimum"
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass_partial_fail_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
