#!/usr/bin/env python3
"""Validate VCAS dataset manifests and reject common data-leakage mistakes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
APPROVED_SOURCE_STATES = {"approved_noncommercial", "approved_internal"}
VALID_SPLITS = {"train", "validation", "test"}
VALID_TASKS = {"detection", "attributes", "multitask"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _is_safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and ":" not in path.parts[0]


def _validate_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def validate_manifest(manifest: dict[str, Any], labels: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required_root = {
        "schema_version",
        "dataset_version",
        "labels_version",
        "created_at",
        "purpose",
        "license_review_status",
        "split_policy",
        "sources",
        "samples",
    }
    missing_root = sorted(required_root - set(manifest))
    if missing_root:
        errors.append(f"root is missing fields: {missing_root}")
        return errors

    if manifest["schema_version"] != "1.0":
        errors.append("schema_version must be 1.0")
    if manifest["labels_version"] != labels.get("labels_version"):
        errors.append("manifest labels_version does not match the canonical label file")
    if manifest["purpose"] not in VALID_TASKS:
        errors.append(f"unsupported dataset purpose: {manifest['purpose']!r}")

    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict):
        errors.append("split_policy must be an object")
        return errors
    group_keys = split_policy.get("group_keys")
    if group_keys != ["camera_id", "video_id", "track_group"]:
        errors.append("group_keys must freeze camera_id, video_id, and track_group in that order")
    ratios = split_policy.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != VALID_SPLITS:
        errors.append("split ratios must define train, validation, and test")
    else:
        try:
            ratio_total = sum(float(ratios[name]) for name in VALID_SPLITS)
        except (TypeError, ValueError):
            errors.append("split ratios must be numeric")
        else:
            if abs(ratio_total - 1.0) > 1e-9:
                errors.append(f"split ratios must sum to 1.0, got {ratio_total}")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty array")
        return errors
    source_map: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        prefix = f"sources[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{prefix} must be an object")
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}.source_id must be a non-empty string")
            continue
        if source_id in source_map:
            errors.append(f"duplicate source_id: {source_id}")
        source_map[source_id] = source
        if source.get("usage_status") not in {
            "approved_noncommercial",
            "approved_internal",
            "review_required",
            "blocked",
        }:
            errors.append(f"{prefix}.usage_status is invalid")
        if source.get("source_type") == "authorized_camera" and not source.get("authorization_ref"):
            errors.append(f"{prefix} authorized camera source requires authorization_ref")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty array")
        return errors

    sample_ids: set[str] = set()
    relative_paths: set[str] = set()
    group_split: dict[tuple[str, str], str] = {}
    vehicle_classes = set(labels.get("vehicle_classes", []))
    body_types = set(labels.get("body_types", []))
    colors = set(labels.get("colors", []))
    crop_qualities = set(labels.get("crop_qualities", []))
    viewpoints = set(labels.get("viewpoints", []))

    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}.sample_id must be a non-empty string")
        elif sample_id in sample_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        else:
            sample_ids.add(sample_id)

        relative_path = sample.get("relative_path")
        if not _is_safe_relative_path(relative_path):
            errors.append(f"{prefix}.relative_path must be a safe POSIX relative path")
        elif relative_path in relative_paths:
            errors.append(f"duplicate relative_path: {relative_path}")
        else:
            relative_paths.add(relative_path)

        if not isinstance(sample.get("sha256"), str) or not SHA256_RE.fullmatch(sample["sha256"]):
            errors.append(f"{prefix}.sha256 must be 64 lowercase hexadecimal characters")

        split = sample.get("split")
        if split not in VALID_SPLITS:
            errors.append(f"{prefix}.split is invalid")
        task = sample.get("task")
        if task not in VALID_TASKS:
            errors.append(f"{prefix}.task is invalid")

        source_id = sample.get("source_id")
        source = source_map.get(source_id)
        if source is None:
            errors.append(f"{prefix}.source_id does not reference a declared source")
        elif source.get("usage_status") not in APPROVED_SOURCE_STATES:
            errors.append(
                f"{prefix} uses source {source_id!r} before license/authorization approval"
            )

        group = sample.get("group")
        if not isinstance(group, dict):
            errors.append(f"{prefix}.group must be an object")
        else:
            for key in ("camera_id", "video_id", "track_group"):
                value = group.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(f"{prefix}.group.{key} must be a non-empty string")
                    continue
                identity = (key, value)
                previous_split = group_split.get(identity)
                if previous_split is not None and previous_split != split:
                    errors.append(
                        f"data leakage: {key}={value!r} appears in "
                        f"{previous_split!r} and {split!r}"
                    )
                else:
                    group_split[identity] = split

        annotations = sample.get("annotations")
        if not isinstance(annotations, dict):
            errors.append(f"{prefix}.annotations must be an object")
            continue
        detection = annotations.get("detection")
        attributes = annotations.get("attributes")
        if task in {"detection", "multitask"} and not isinstance(detection, dict):
            errors.append(f"{prefix} task {task!r} requires detection annotations")
        if task in {"attributes", "multitask"} and not isinstance(attributes, dict):
            errors.append(f"{prefix} task {task!r} requires attribute annotations")

        if isinstance(detection, dict):
            if not _validate_bbox(detection.get("bbox_xyxy_norm")):
                errors.append(f"{prefix}.annotations.detection bbox is invalid")
            if detection.get("vehicle_class") not in vehicle_classes:
                errors.append(f"{prefix}.annotations.detection vehicle_class is invalid")

        if isinstance(attributes, dict):
            if attributes.get("body_type") not in body_types:
                errors.append(f"{prefix}.annotations.attributes body_type is invalid")
            if attributes.get("color") not in colors:
                errors.append(f"{prefix}.annotations.attributes color is invalid")
            if attributes.get("crop_quality") not in crop_qualities:
                errors.append(f"{prefix}.annotations.attributes crop_quality is invalid")
            if attributes.get("viewpoint") not in viewpoints:
                errors.append(f"{prefix}.annotations.attributes viewpoint is invalid")
            if attributes.get("crop_quality") == "poor":
                if attributes.get("body_type") != "unknown" or attributes.get("color") != "unknown":
                    errors.append(
                        f"{prefix} poor crop must use unknown body_type and color"
                    )

    if manifest.get("license_review_status") == "approved":
        unapproved = [
            source_id
            for source_id, source in source_map.items()
            if source.get("usage_status") not in APPROVED_SOURCE_STATES
        ]
        if unapproved:
            errors.append(
                "license_review_status is approved while sources remain unapproved: "
                + ", ".join(sorted(unapproved))
            )

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("config/vehicle_labels.v1.json"),
        help="canonical label mapping",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = load_json(args.manifest)
        labels = load_json(args.labels)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(manifest, labels)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: {args.manifest} "
        f"dataset_version={manifest['dataset_version']} samples={len(manifest['samples'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
