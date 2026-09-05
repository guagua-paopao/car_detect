#!/usr/bin/env python3
"""Append Stage241's agent-reviewed night tracks to the Stage157 color manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--stage241-manifest", required=True, type=Path)
    parser.add_argument("--expected-stage241-sha256", required=True)
    parser.add_argument("--stage241-report", required=True, type=Path)
    parser.add_argument("--expected-stage241-report-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output_dir}")
    for path, expected in (
        (args.base_manifest, args.expected_base_sha256),
        (args.stage241_manifest, args.expected_stage241_sha256),
        (args.stage241_report, args.expected_stage241_report_sha256),
    ):
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"SHA256 mismatch: {path}: {actual}")
    stage241_report = json.loads(args.stage241_report.read_text(encoding="utf-8"))
    if stage241_report.get("status") != "complete_targeted_research_candidate_authorized":
        raise RuntimeError("Stage241 did not authorize the targeted research candidate")
    if stage241_report.get("quota", {}).get("pass") is not True:
        raise RuntimeError("Stage241 quota did not pass")

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        base_rows = list(reader)
    if not base_fields or not base_rows:
        raise RuntimeError("base manifest is empty")
    with args.stage241_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        night_rows = list(csv.DictReader(handle))
    if len(night_rows) != 264:
        raise RuntimeError(f"expected 264 Stage241 rows, got {len(night_rows)}")
    if any(row.get("split") != "train" for row in night_rows):
        raise RuntimeError("Stage241 contains non-train rows")
    if any(not truthy(row.get("color_supervised", "false")) for row in night_rows):
        raise RuntimeError("Stage241 contains unsupervised rows")
    if any(row.get("color") == "unknown" for row in night_rows):
        raise RuntimeError("Stage241 contains unknown supervised colors")
    if any(row.get("cross_source_duplicate_of", "").strip() for row in night_rows):
        raise RuntimeError("Stage241 contains a cross-source duplicate")
    if any(row.get("internal_near_duplicate_of", "").strip() for row in night_rows):
        raise RuntimeError("Stage241 contains a selected internal near duplicate")

    base_split_counts = Counter(row["split"] for row in base_rows)
    base_train_colors = Counter(row["color"] for row in base_rows if row["split"] == "train" and truthy(row.get("color_supervised", "false")))
    base_validation_colors = Counter(row["color"] for row in base_rows if row["split"] == "validation" and truthy(row.get("color_supervised", "false")))
    base_hashes = {
        (row.get("crop_sha256") or row.get("sha256") or "").lower()
        for row in base_rows
        if row.get("crop_sha256") or row.get("sha256")
    }
    base_dhashes = {
        (row.get("crop_dhash64") or row.get("dhash64") or "").lower()
        for row in base_rows
        if row.get("crop_dhash64") or row.get("dhash64")
    }
    exact_overlap = sum(row["crop_sha256"].lower() in base_hashes for row in night_rows)
    dhash_overlap = sum(row["crop_dhash64"].lower() in base_dhashes for row in night_rows)
    if exact_overlap or dhash_overlap:
        raise RuntimeError(f"cross-source overlap detected exact={exact_overlap} dhash={dhash_overlap}")

    recording_row_counts = Counter(row["stage241_recording_group"] for row in night_rows)
    normalized_rows = []
    for row in night_rows:
        out = {field: "" for field in base_fields}
        values = {
            "image_path": row["crop_path"],
            "body_type": "unknown",
            "color": row["color"],
            "crop_quality": "good",
            "viewpoint": "unknown",
            "blur": "false",
            "occluded": "false",
            "truncated": row["border_truncated"],
            "night": "true",
            "camera_id": f"NightOwls-recording-{int(row['recording_id']):02d}",
            "video_id": row["stage241_recording_group"],
            "track_group": row["track_key"],
            "split": "train",
            "source_frame_id": row["source_image_id"],
            "review_status": "approved_agent_three_frame_visible_color",
            "body_type_supervised": "false",
            "color_supervised": "true",
            "source_dataset": "NightOwls",
            "source_color": row["color"],
            "source_license": "non-commercial research-only; no redistribution",
            "review_method": "agent_visual_three_temporally_separated_frames+recording_color_cap",
            "sha256": row["crop_sha256"],
            "dhash64": row["crop_dhash64"],
            "width": row["crop_width"],
            "height": row["crop_height"],
            "gray_mean": row["crop_mean_luma"],
            "gray_stddev": row["crop_contrast"],
            "edge_variance": row["crop_sharpness"],
            "source_group": row["stage241_recording_group"],
            "source_group_images": str(recording_row_counts[row["stage241_recording_group"]]),
            "source_sha256": row["source_frame_sha256"],
            "crop_sha256": row["crop_sha256"],
            "crop_dhash64": row["crop_dhash64"],
            "taxonomy_v2_color_source": "agent_reviewed_night_track",
            "taxonomy_v2_eligibility": "research-only_non-deployable",
            "annotation_source": "stage241_agent_visual_track_consensus",
            "occlusion_level": "unknown",
            "vehicle_size": row["size_bin"],
            "lighting": "night",
            "label_confidence": "0.95",
            "license_train_eligible": "true",
            "color_review_status": "agent_visual_three_frame_consensus",
            "review_score": row["stage240_average_quality"],
            "review_model": "codex_agent_visual_review",
            "formal_train_eligible": "true",
            "hard_example_priority": "0.80",
            "hard_mining_tags": "real-night;visible-spectrum;multiframe;research-only",
            "hard_score": "0.80",
            "photometric_mean": row["crop_mean_luma"],
            "photometric_contrast": row["crop_contrast"],
            "photometric_sharpness": row["crop_sharpness"],
            "pseudo_label": "false",
            "pseudo_label_confidence": "0",
            "source_image_id": row["source_image_id"],
            "small_target": str(row["size_bin"] == "small").lower(),
            "low_light": "true",
            "crop_mean_luma": row["crop_mean_luma"],
            "crop_sharpness": row["crop_sharpness"],
            "sample_weight": "1.0",
            "training_reason": "targeted_agent_reviewed_real_night_color_repair",
            "weather": "night",
            "source_image_sha256": row["source_frame_sha256"],
            "research_only": "true",
            "training_eligible": "true",
            "usage": "train_research_only",
            "source_split": "train",
            "source_video": row["stage241_recording_group"],
            "track_id": row["stage240_audit_id"],
            "track_key": row["track_key"],
            "source_box_width": row["crop_width"],
            "source_box_height": row["crop_height"],
            "frame_mean_luma": row["crop_mean_luma"],
            "crop_contrast": row["crop_contrast"],
        }
        for key, value in values.items():
            if key in out:
                out[key] = str(value)
        normalized_rows.append(out)

    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "attribute_manifest.stage242-nightowls-color-repair.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(normalized_rows)

    added_colors = Counter(row["color"] for row in normalized_rows)
    combined_train_colors = base_train_colors + added_colors
    report = {
        "schema_version": "stage242-nightowls-color-repair-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_only_targeted_night_repair",
        "inputs": {
            "base_manifest": str(args.base_manifest),
            "base_manifest_sha256": sha256(args.base_manifest),
            "stage241_manifest": str(args.stage241_manifest),
            "stage241_manifest_sha256": sha256(args.stage241_manifest),
            "stage241_report": str(args.stage241_report),
            "stage241_report_sha256": sha256(args.stage241_report),
        },
        "base": {
            "rows": len(base_rows),
            "split_counts": dict(sorted(base_split_counts.items())),
            "train_color_counts": dict(sorted(base_train_colors.items())),
            "validation_color_counts": dict(sorted(base_validation_colors.items())),
        },
        "added": {
            "rows": len(normalized_rows),
            "tracks": len({row["track_key"] for row in night_rows}),
            "recording_groups": len(recording_row_counts),
            "color_counts": dict(sorted(added_colors.items())),
            "all_train_split": True,
            "exact_cross_source_overlap": exact_overlap,
            "exact_dhash_cross_source_overlap": dhash_overlap,
            "prior_stage234_cross_source_near_distance": 2,
            "prior_stage234_cross_source_duplicates": 0,
        },
        "output": {
            "manifest": str(output_path),
            "manifest_sha256": sha256(output_path),
            "rows": len(base_rows) + len(normalized_rows),
            "train_rows": base_split_counts["train"] + len(normalized_rows),
            "validation_rows": base_split_counts["validation"],
            "train_color_counts": dict(sorted(combined_train_colors.items())),
            "validation_color_counts": dict(sorted(base_validation_colors.items())),
        },
        "training_policy": {
            "initialization": "Stage158 ConvNeXt-Tiny 256 best checkpoint",
            "night_rows_are_targeted_low_weight_repair_not_standalone_training": True,
            "retain_full_existing_supervised_replay": True,
            "maximum_tracks_per_recording_and_color": 2,
            "group_night_samples_by_recording": True,
            "research_only": True,
            "deployment_eligible": False,
        },
        "policy": {
            "validation_rows_unchanged": True,
            "test_rows_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage242-nightowls-color-repair-manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SHA256SUMS").write_text(
        f"{sha256(output_path)}  {output_path.name}\n{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
