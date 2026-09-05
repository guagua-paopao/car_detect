#!/usr/bin/env python3
"""Build a compact train-only union for the current full scene pixel audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FROZEN_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}
FIELDS = [
    "image_path", "split", "review_status", "body_type", "color",
    "body_type_supervised", "color_supervised", "formal_train_eligible",
    "coarse_body_family", "coarse_color_group", "source_dataset", "source_group",
    "camera_id", "video_id", "track_group", "source_frame_id", "source_image_id",
    "lighting", "night", "low_light", "weather", "source_manifest",
    "source_license", "license_train_eligible", "source_partition",
    "stage214_input_role",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_input(path: Path, expected_sha256: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"input must be a regular non-symlink file: {resolved}")
    actual = sha256_file(resolved)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"input SHA256 mismatch for {resolved}: {actual}")
    return actual


def iter_train_rows(path: Path, role: str, allowed_root: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split", "review_status", "body_type", "color"}
        if missing := required - set(reader.fieldnames or []):
            raise RuntimeError(f"{role} manifest missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            if str(row.get("split") or "").strip().lower() != "train":
                continue
            searchable = " ".join(str(row.get(field) or "") for field in (
                "image_path", "source_manifest", "source_frame_id", "video_id",
            )).lower()
            if any(marker in searchable for marker in FROZEN_MARKERS):
                raise RuntimeError(f"frozen-video marker in {role} row {row_number}")
            image = Path(str(row.get("image_path") or ""))
            if not image.is_absolute():
                image = (path.parent / image).resolve()
            else:
                image = image.resolve()
            try:
                image.relative_to(allowed_root)
            except ValueError as error:
                raise RuntimeError(f"image outside allowed root in {role} row {row_number}: {image}") from error
            output = {field: str(row.get(field) or "") for field in FIELDS}
            output["image_path"] = str(image)
            output["split"] = "train"
            output["stage214_input_role"] = role
            yield output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-manifest", type=Path, required=True)
    parser.add_argument("--expected-supervised-sha256", required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--expected-unlabeled-sha256", required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    allowed_root = args.allowed_root.resolve()
    supervised = args.supervised_manifest.resolve()
    unlabeled = args.unlabeled_manifest.resolve()
    supervised_sha = validate_input(supervised, args.expected_supervised_sha256)
    unlabeled_sha = validate_input(unlabeled, args.expected_unlabeled_sha256)

    args.output_root.mkdir(parents=True, exist_ok=False)
    union_path = args.output_root / "attribute_manifest.stage214-current-train-scene-union.csv"
    counts: Counter[str] = Counter()
    paths_by_role: dict[str, set[str]] = {"supervised": set(), "unlabeled": set()}
    with union_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for role, path in (("supervised", supervised), ("unlabeled", unlabeled)):
            for row in iter_train_rows(path, role, allowed_root):
                writer.writerow(row)
                counts[role] += 1
                paths_by_role[role].add(row["image_path"])

    if not counts["supervised"] or not counts["unlabeled"]:
        raise RuntimeError("both current train roles must contain rows")
    union_sha = sha256_file(union_path)
    report = {
        "schema_version": "stage214-current-train-scene-union-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_compact_train_only_union_ready",
        "inputs": {
            "supervised_manifest": str(supervised),
            "supervised_manifest_sha256": supervised_sha,
            "unlabeled_manifest": str(unlabeled),
            "unlabeled_manifest_sha256": unlabeled_sha,
        },
        "counts": {
            "supervised_train_rows": counts["supervised"],
            "unlabeled_train_rows": counts["unlabeled"],
            "union_rows": sum(counts.values()),
            "supervised_unique_paths": len(paths_by_role["supervised"]),
            "unlabeled_unique_paths": len(paths_by_role["unlabeled"]),
            "cross_role_path_overlap": len(paths_by_role["supervised"] & paths_by_role["unlabeled"]),
            "combined_unique_paths": len(paths_by_role["supervised"] | paths_by_role["unlabeled"]),
            "validation_or_test_rows_written": 0,
        },
        "outputs": {"union_manifest": str(union_path), "union_manifest_sha256": union_sha},
        "policy": {
            "metadata_only": True,
            "source_manifests_immutable": True,
            "attribute_labels_unchanged": True,
            "train_partition_only": True,
            "validation_or_test_pixels_opened": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage214-current-train-scene-union-report.json"
    atomic_json(report_path, report)
    for path in (union_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
