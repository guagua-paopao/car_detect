#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, str, int]:
    return (
        row.get("video_id", "").strip(),
        row.get("track_group", "").strip(),
        int(row.get("frame_number", "0") or 0),
    )


def choose_spread(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: int(row["frame_number"]))
    if count <= 0:
        return []
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = []
    for position in range(count):
        index = round(position * (len(ordered) - 1) / (count - 1))
        if index not in indexes:
            indexes.append(index)
    return [ordered[index] for index in indexes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage70-root", required=True, type=Path)
    parser.add_argument("--stage177-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--combined-track-cap", type=int, default=5)
    args = parser.parse_args()

    stage70_root = args.stage70_root.resolve()
    stage70_manifest = (stage70_root / "attribute_manifest.training-only.csv").resolve()
    stage70_report = (stage70_root / "dataset_report.json").resolve()
    stage177_manifest = args.stage177_manifest.resolve()
    output_root = args.output_root.resolve()
    for path in (stage70_root, stage70_manifest, stage70_report, stage177_manifest, output_root):
        lowered = str(path).lower()
        if "vcas_rtsp_demo_60s" in lowered or "36-48" in lowered or "/test" in lowered or "\\test" in lowered:
            raise RuntimeError(f"prohibited path marker: {path}")
    if not 1 <= args.combined_track_cap <= 5:
        raise RuntimeError("combined track cap must be in [1, 5]")
    for path in (stage70_manifest, stage70_report, stage177_manifest):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked input: {path}")
    output_root.mkdir(parents=True, exist_ok=False)

    with stage70_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        stage70_fields = list(reader.fieldnames or [])
        stage70_rows = list(reader)
    stage70_keys = [key(row) for row in stage70_rows]
    if len(stage70_keys) != len(set(stage70_keys)):
        raise RuntimeError("Stage70 contains duplicate video/track/frame keys")
    stage70_by_key = dict(zip(stage70_keys, stage70_rows))

    stage177_all_rows = []
    with stage177_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        stage177_all_rows = list(csv.DictReader(stream))
    stage177_rows = [
        row
        for row in stage177_all_rows
        if row.get("source_dataset", "").strip().upper() == "UA-DETRAC"
    ]
    stage177_keys = {key(row) for row in stage177_rows}
    overlap_keys = set(stage70_by_key) & stage177_keys
    stage177_not_in_stage70 = stage177_keys - set(stage70_by_key)
    missing_rows = [row for row in stage70_rows if key(row) not in stage177_keys]
    stage70_official_night_keys = {
        key(row) for row in stage70_rows if row.get("weather", "").strip().lower() == "night"
    }

    existing_effective_by_track = Counter()
    for row in stage177_rows:
        if row.get("stage177_effective_representative", "").strip().lower() == "true":
            existing_effective_by_track[row["track_group"].strip()] += 1
    missing_night_by_track: dict[str, list[dict]] = defaultdict(list)
    for row in missing_rows:
        if row.get("weather", "").strip().lower() == "night":
            missing_night_by_track[row["track_group"].strip()].append(row)

    selected = []
    tracks_at_or_above_cap = 0
    for track, candidates in sorted(missing_night_by_track.items()):
        existing = existing_effective_by_track[track]
        remaining = max(0, args.combined_track_cap - existing)
        if remaining == 0:
            tracks_at_or_above_cap += 1
            continue
        for row in choose_spread(candidates, remaining):
            output = dict(row)
            output["stage205_gap_reason"] = "absent_from_stage177_loader_manifest"
            output["stage205_existing_stage177_effective_rows_for_track"] = str(existing)
            output["stage205_combined_track_cap"] = str(args.combined_track_cap)
            selected.append(output)

    missing_crop_files = []
    for row in selected:
        crop_path = (stage70_root / row["image_path"]).resolve()
        if stage70_root not in crop_path.parents or not crop_path.is_file() or crop_path.is_symlink():
            missing_crop_files.append(str(crop_path))
    if missing_crop_files:
        raise RuntimeError(f"selected Stage70 crop files missing or unsafe: {missing_crop_files[:5]}")

    stage177_effective_rows = [
        row
        for row in stage177_all_rows
        if row.get("stage177_effective_representative", "").strip().lower() == "true"
    ]
    stage177_scene_night_effective = [
        row
        for row in stage177_effective_rows
        if row.get("stage177_scene_label", "").strip().lower() == "night"
    ]
    stage177_scene_lowlight_effective = [
        row
        for row in stage177_effective_rows
        if row.get("stage177_scene_label", "").strip().lower() == "low_light"
    ]
    stage177_uadetrac_official_night_effective = [
        row
        for row in stage177_rows
        if row.get("stage177_effective_representative", "").strip().lower() == "true"
        and key(row) in stage70_official_night_keys
    ]
    stage177_harmonized_night_effective = [
        row
        for row in stage177_effective_rows
        if row.get("stage177_scene_label", "").strip().lower() == "night"
        or (
            row.get("source_dataset", "").strip().upper() == "UA-DETRAC"
            and key(row) in stage70_official_night_keys
        )
    ]
    projected_effective_rows = len(stage177_effective_rows) + len(selected)
    projected_harmonized_night_rows = len(stage177_harmonized_night_effective) + len(selected)

    output_fields = stage70_fields + [
        "stage205_gap_reason",
        "stage205_existing_stage177_effective_rows_for_track",
        "stage205_combined_track_cap",
    ]
    manifest = output_root / "stage205-uadetrac-night-gap-track-cap5.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: (row["split"], row["video_id"], row["track_group"], int(row["frame_number"]))))

    report = {
        "schema_version": "stage205-uadetrac-stage177-gap-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_gap_quantified_no_training",
        "inputs": {
            "stage70_manifest": str(stage70_manifest),
            "stage70_manifest_sha256": sha256_path(stage70_manifest),
            "stage70_report": str(stage70_report),
            "stage70_report_sha256": sha256_path(stage70_report),
            "stage177_manifest": str(stage177_manifest),
            "stage177_manifest_sha256": sha256_path(stage177_manifest),
        },
        "comparison": {
            "stage70_rows": len(stage70_rows),
            "stage177_uadetrac_rows": len(stage177_rows),
            "overlap_keys": len(overlap_keys),
            "stage177_keys_not_in_stage70": len(stage177_not_in_stage70),
            "stage70_rows_absent_from_stage177": len(missing_rows),
            "stage70_official_night_rows": len(stage70_official_night_keys),
            "stage70_official_night_rows_present_in_stage177": sum(
                key(row) in stage177_keys
                for row in stage70_rows
                if row.get("weather", "").strip().lower() == "night"
            ),
            "stage70_missing_rows_by_weather": dict(sorted(Counter(row["weather"] for row in missing_rows).items())),
            "stage70_missing_night_rows_by_source_type": dict(sorted(Counter(row["source_vehicle_type"] for row in missing_rows if row["weather"] == "night").items())),
            "stage70_missing_night_rows_by_body_label": dict(sorted(Counter(row["body_type"] for row in missing_rows if row["weather"] == "night").items())),
            "stage70_missing_night_rows_by_split": dict(sorted(Counter(row["split"] for row in missing_rows if row["weather"] == "night").items())),
        },
        "combined_track_cap5_supplement": {
            "existing_stage177_effective_night_rows": sum(
                row.get("stage177_effective_representative", "").strip().lower() == "true" and row.get("stage177_scene_label", "").strip().lower() == "night"
                for row in stage177_rows
            ),
            "tracks_with_missing_night_rows": len(missing_night_by_track),
            "tracks_already_at_or_above_cap": tracks_at_or_above_cap,
            "selected_supplement_rows": len(selected),
            "selected_by_split": dict(sorted(Counter(row["split"] for row in selected).items())),
            "selected_by_source_type": dict(sorted(Counter(row["source_vehicle_type"] for row in selected).items())),
            "selected_exact_body_rows": dict(sorted(Counter(row["body_type"] for row in selected if row["body_type_supervised"] == "true").items())),
            "selected_coarse_body_rows": dict(sorted(Counter(row["coarse_body_family"] for row in selected if row["coarse_body_family"]).items())),
        },
        "effective_night_recalculation": {
            "stage177_all_effective_rows": len(stage177_effective_rows),
            "stage177_scene_night_effective_rows": len(stage177_scene_night_effective),
            "stage177_scene_lowlight_effective_rows": len(stage177_scene_lowlight_effective),
            "stage177_uadetrac_official_night_effective_rows": len(stage177_uadetrac_official_night_effective),
            "stage177_harmonized_night_effective_rows": len(stage177_harmonized_night_effective),
            "stage177_harmonized_night_share": (
                len(stage177_harmonized_night_effective) / len(stage177_effective_rows)
                if stage177_effective_rows
                else 0.0
            ),
            "projected_effective_rows_if_all_supplements_pass_pixel_audit": projected_effective_rows,
            "projected_harmonized_night_rows_if_all_supplements_pass_pixel_audit": projected_harmonized_night_rows,
            "projected_harmonized_night_share_if_all_supplements_pass_pixel_audit": (
                projected_harmonized_night_rows / projected_effective_rows
                if projected_effective_rows
                else 0.0
            ),
        },
        "outputs": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_path(manifest),
        },
        "policy": {
            "selected_rows_are_existing_stage70_crops_not_new_copies": True,
            "license_status": "research-only_non-deployable; mirror declares CC BY 4.0 but upstream legal review remains required",
            "training_authorized": False,
            "pixel_reaudit_required": True,
            "cross_source_near_duplicate_audit_required": True,
            "teacher_consensus_required_for_coarse_car_fine_labels": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = output_root / "stage205-uadetrac-stage177-gap.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "SHA256SUMS").write_text(
        f"{sha256_path(report_path)}  {report_path.name}\n"
        f"{sha256_path(manifest)}  {manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
