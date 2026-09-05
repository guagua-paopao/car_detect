#!/usr/bin/env python3
"""Merge exact DVM v2 color truth with training-only CCTV partial color truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PARTIAL_COLOR_GROUPS = {"silver_gray", "yellow_orange", "brown_beige"}
EXACT_SHARED_COLORS = {"black", "white", "red", "blue", "green", "other"}
TRUE_VALUES = {"true", "1", "yes"}


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def is_training_only_cctv(row: dict[str, str]) -> bool:
    return (
        row.get("split") == "train"
        and row.get("review_status") == "approved"
        and truthy(row.get("formal_train_eligible"))
        and truthy(row.get("license_train_eligible"))
        and bool(row.get("source_license", "").strip())
        and truthy(row.get("color_supervised"))
        and row.get("color") not in {"", "unknown"}
        and (
            row.get("stage71_origin") == "cctv"
            or bool(row.get("stage74_joint_source", "").strip())
        )
    )


def convert_cctv_row(row: dict[str, str]) -> dict[str, str] | None:
    if not is_training_only_cctv(row):
        return None
    output = dict(row)
    source_color = row.get("color", "").strip().lower()
    output["body_type"] = "unknown"
    output["body_type_supervised"] = "false"
    output["coarse_body_family"] = ""
    output["coarse_color_group"] = ""
    if source_color in EXACT_SHARED_COLORS:
        output["color"] = source_color
        output["color_supervised"] = "true"
        supervision = "exact_v2_shared_color"
    elif source_color in PARTIAL_COLOR_GROUPS:
        output["color"] = "unknown"
        output["color_supervised"] = "false"
        output["coarse_color_group"] = source_color
        supervision = "partial_v2_color_probability_mass"
    else:
        return None
    output["split"] = "train"
    output["pseudo_label"] = "true"
    output["stage102_origin"] = "stage74_training_only_cctv"
    output["stage102_source_color"] = source_color
    output["stage102_supervision"] = supervision
    output["stage102_policy"] = "no_exact_class_fabrication_for_legacy_merged_color"
    return output


def row_key(row: dict[str, str]) -> str:
    return row.get("sha256", "").strip().lower() or row.get("crop_sha256", "").strip().lower() or row.get("image_path", "")


def group_keys(row: dict[str, str]) -> set[str]:
    return {
        value
        for value in (
            row.get("source_group", "").strip(),
            row.get("track_group", "").strip(),
            row.get("video_id", "").strip(),
        )
        if value
    }


def parse_dhash(row: dict[str, str]) -> int | None:
    value = (row.get("dhash64") or row.get("crop_dhash64") or "").strip().lower()
    if not value:
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def near_duplicate(value: int, validation_hashes: list[int], distance: int) -> bool:
    return any((value ^ candidate).bit_count() <= distance for candidate in validation_hashes)


def resolve_image(path_text: str, manifest: Path, safety_root: Path) -> Path:
    candidate = Path(path_text)
    resolved = candidate.resolve() if candidate.is_absolute() else (manifest.parent / candidate).resolve()
    try:
        resolved.relative_to(safety_root.resolve())
    except ValueError as error:
        raise ValueError(f"image escapes safety root: {resolved}") from error
    return resolved


def image_is_readable(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def build(
    base_manifest: Path,
    cctv_manifest: Path,
    safety_root: Path,
    near_distance: int,
    audit_images: bool,
    minimum_cctv_rows: int = 1500,
    minimum_groups: int = 5,
) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    base_fields, base_rows = load_rows(base_manifest)
    cctv_fields, cctv_rows = load_rows(cctv_manifest)
    if not base_rows or not any(row.get("split") == "validation" for row in base_rows):
        raise ValueError("base v2 manifest must contain an immutable validation split")
    validation_rows = [row for row in base_rows if row.get("split") == "validation"]
    validation_exact = {row_key(row) for row in validation_rows if row_key(row)}
    validation_groups = set().union(*(group_keys(row) for row in validation_rows))
    validation_dhashes = [value for row in validation_rows if (value := parse_dhash(row)) is not None]
    seen_exact = {row_key(row) for row in base_rows if row_key(row)}
    accepted: list[dict[str, str]] = []
    rejection_counts: Counter[str] = Counter()

    for source in cctv_rows:
        converted = convert_cctv_row(source)
        if converted is None:
            continue
        key = row_key(converted)
        if not key:
            rejection_counts["missing_exact_identity"] += 1
            continue
        if key in validation_exact or key in seen_exact:
            rejection_counts["exact_duplicate"] += 1
            continue
        if group_keys(converted) & validation_groups:
            rejection_counts["group_leak"] += 1
            continue
        dhash = parse_dhash(converted)
        if dhash is not None and near_duplicate(dhash, validation_dhashes, near_distance):
            rejection_counts["validation_near_duplicate"] += 1
            continue
        if audit_images:
            try:
                image_path = resolve_image(converted.get("image_path", ""), cctv_manifest, safety_root)
            except ValueError:
                rejection_counts["unsafe_image_path"] += 1
                continue
            if not image_is_readable(image_path):
                rejection_counts["unreadable_image"] += 1
                continue
            converted["image_path"] = str(image_path)
        seen_exact.add(key)
        accepted.append(converted)

    if len(accepted) < minimum_cctv_rows:
        raise ValueError(
            f"only {len(accepted)} eligible CCTV color rows; require at least {minimum_cctv_rows}"
        )
    supervision_counts = Counter(
        row.get("coarse_color_group") or row.get("color") for row in accepted
    )
    if len(supervision_counts) < minimum_groups:
        raise ValueError(
            f"only {len(supervision_counts)} CCTV supervision groups; require at least {minimum_groups}"
        )

    fields = list(dict.fromkeys([
        *base_fields,
        *cctv_fields,
        "coarse_color_group",
        "stage102_origin",
        "stage102_source_color",
        "stage102_supervision",
        "stage102_policy",
    ]))
    output_rows = [dict(row) for row in base_rows] + accepted
    report = {
        "schema_version": "stage102-v2-partial-color-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "inputs": {
            "base_v2_manifest": str(base_manifest.resolve()),
            "base_v2_sha256": sha256(base_manifest),
            "stage74_manifest": str(cctv_manifest.resolve()),
            "stage74_sha256": sha256(cctv_manifest),
        },
        "rows": {
            "base": len(base_rows),
            "base_train": sum(row.get("split") == "train" for row in base_rows),
            "base_validation": len(validation_rows),
            "added_cctv_train": len(accepted),
            "output": len(output_rows),
        },
        "added_supervision_counts": dict(sorted(supervision_counts.items())),
        "added_source_counts": dict(sorted(Counter(row.get("source_dataset", "unknown") for row in accepted).items())),
        "added_license_counts": dict(sorted(Counter(row.get("source_license", "unknown") for row in accepted).items())),
        "added_annotation_source_counts": dict(sorted(Counter(row.get("annotation_source", "unknown") for row in accepted).items())),
        "added_scene_counts": {
            "night": sum(truthy(row.get("night")) for row in accepted),
            "low_light": sum(truthy(row.get("low_light")) or "low_light" in row.get("lighting", "") for row in accepted),
            "small": sum(truthy(row.get("small_target")) or row.get("vehicle_size") == "small" for row in accepted),
        },
        "rejections": dict(sorted(rejection_counts.items())),
        "leakage": {
            "exact_train_validation": 0,
            "group_train_validation": 0,
            "near_distance": near_distance,
            "near_train_validation": 0,
        },
        "policy": {
            "test_accessed": False,
            "frozen_video_used": False,
            "validation_rows_added_to_training": 0,
            "cctv_non_train_rows_selected": 0,
            "missing_or_ineligible_license_rows_selected": 0,
            "missing_or_unapproved_review_rows_selected": 0,
            "merged_silver_gray_exact_class_fabricated": False,
            "base_v2_readability_and_near_duplicate_audit_inherited_from_pinned_parent": True,
            "stage74_cctv_readability_and_near_duplicate_audit_inherited_from_pinned_parent": True,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    return fields, output_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v2-manifest", type=Path, required=True)
    parser.add_argument("--stage74-manifest", type=Path, required=True)
    parser.add_argument("--safety-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-stage74-sha256", required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=2)
    parser.add_argument("--skip-image-audit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_manifest = args.base_v2_manifest.resolve()
    stage74_manifest = args.stage74_manifest.resolve()
    if sha256(base_manifest).lower() != args.expected_base_sha256.lower():
        raise ValueError("base v2 manifest SHA256 mismatch")
    if sha256(stage74_manifest).lower() != args.expected_stage74_sha256.lower():
        raise ValueError("Stage74 manifest SHA256 mismatch")
    if not 0 <= args.near_duplicate_hamming <= 8:
        raise ValueError("near duplicate Hamming distance must be in [0, 8]")
    fields, rows, report = build(
        base_manifest,
        stage74_manifest,
        args.safety_root,
        args.near_duplicate_hamming,
        not args.skip_image_audit,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "attribute_manifest.stage102-v2-partial-color.csv"
    report_path = output_dir / "stage102-v2-partial-color-report.json"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report["output"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
