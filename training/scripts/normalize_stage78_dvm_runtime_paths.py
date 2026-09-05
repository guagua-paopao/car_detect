#!/usr/bin/env python3
"""Create a decode- and SHA-audited runtime-safe Stage78 manifest."""

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


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def projection_update(digest: Any, row: dict[str, str], fields: list[str]) -> None:
    payload = [str(row.get(field, "")) for field in fields if field != "image_path"]
    digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def verify_image(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage78 runtime evidence: {path}")
    pinned = (
        (args.input_manifest, args.expected_input_manifest_sha256),
        (args.input_report, args.expected_input_report_sha256),
        (args.labels, args.expected_labels_sha256),
    )
    for path, expected in pinned:
        if not path.is_file() or sha256(path).lower() != expected.lower():
            raise RuntimeError(f"immutable Stage78 input mismatch: {path}")

    report = read_json(args.input_report)
    labels = read_json(args.labels)
    if report.get("status") != "pass":
        raise RuntimeError("Stage78 input report did not pass")
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 labels are required")
    if report.get("output", {}).get("sha256", "").lower() != sha256(args.input_manifest):
        raise RuntimeError("Stage78 report does not bind its manifest")
    policy = report.get("policy", {})
    for key in ("test_rows_imported", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage78 isolation violation: {key}")

    safety_root = args.datasets_safety_root.resolve()
    source_root = args.source_root.resolve()
    if not safety_root.is_dir() or not source_root.is_dir():
        raise RuntimeError("Stage78 runtime roots are missing")
    try:
        source_root.relative_to(safety_root)
    except ValueError as error:
        raise RuntimeError("Stage78 source root escapes the datasets safety root") from error

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
    counts: Counter[str] = Counter()
    before = hashlib.sha256()
    after = hashlib.sha256()
    allowed_colors = set(labels["colors"])
    try:
        with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            required = {"image_path", "split", "color", "color_supervised", "review_status", "crop_sha256"}
            if not required.issubset(fields):
                raise RuntimeError(f"Stage78 manifest is missing fields: {sorted(required - set(fields))}")
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
                writer.writeheader()
                for line, original in enumerate(reader, start=2):
                    split = original.get("split", "").strip().lower()
                    if split not in {"train", "validation"}:
                        raise RuntimeError(f"forbidden Stage78 split at line {line}: {split}")
                    searchable = " ".join(original.values()).lower()
                    if any(marker in searchable for marker in FROZEN_MARKERS):
                        raise RuntimeError(f"frozen marker in Stage78 row {line}")
                    if original.get("review_status") != "approved":
                        raise RuntimeError(f"unapproved Stage78 row {line}")
                    if original.get("color_supervised", "").strip().lower() not in {"1", "true", "yes"}:
                        raise RuntimeError(f"unsupervised Stage78 row {line}")
                    color = original.get("color", "")
                    if color not in allowed_colors or color == "unknown":
                        raise RuntimeError(f"invalid Stage78 color at line {line}: {color}")
                    raw = Path(original.get("image_path", ""))
                    if not str(raw) or raw.is_absolute():
                        raise RuntimeError(f"Stage78 R1 path must be relative at line {line}")
                    resolved = (source_root / raw).resolve()
                    try:
                        resolved.relative_to(source_root)
                        resolved.relative_to(safety_root)
                    except ValueError as error:
                        raise RuntimeError(f"Stage78 image escapes runtime roots at line {line}: {resolved}") from error
                    if not resolved.is_file() or resolved.stat().st_size <= 0:
                        raise FileNotFoundError(resolved)
                    expected_crop_sha = original.get("crop_sha256", "").strip().lower()
                    actual_crop_sha = sha256(resolved)
                    if not expected_crop_sha or actual_crop_sha != expected_crop_sha:
                        raise RuntimeError(f"Stage78 crop SHA256 mismatch at line {line}")
                    verify_image(resolved)

                    projection_update(before, original, fields)
                    row = dict(original)
                    row["image_path"] = str(resolved)
                    projection_update(after, row, fields)
                    writer.writerow(row)
                    counts["rows"] += 1
                    counts[f"split:{split}"] += 1
                    counts[f"{split}:{color}"] += 1
                    counts["paths:absolute"] += 1
                    counts["decoded"] += 1
                    counts["crop_sha256_verified"] += 1
        if before.hexdigest() != after.hexdigest():
            raise RuntimeError("Stage78 non-path projection changed")
        expected_rows = int(report.get("output", {}).get("rows", -1))
        if counts["rows"] != expected_rows:
            raise RuntimeError(f"Stage78 row count mismatch: {dict(counts)}")
        os.replace(temporary, args.output_manifest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    output_manifest_sha = sha256(args.output_manifest)
    result = {
        "schema_version": "stage78-dvm-fine-color-runtime-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "input": {
            "manifest": str(args.input_manifest.resolve()),
            "manifest_sha256": sha256(args.input_manifest),
            "report": str(args.input_report.resolve()),
            "report_sha256": sha256(args.input_report),
            "source_root": str(source_root),
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256(args.labels),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "rows": counts["rows"],
            "counts": dict(sorted(counts.items())),
            "sha256": output_manifest_sha,
        },
        "integrity": {
            "non_path_projection_sha256": before.hexdigest(),
            "non_path_projection_unchanged": True,
            "all_paths_absolute_within_source_root": True,
            "all_images_decoded": counts["decoded"] == counts["rows"],
            "all_crop_sha256_verified": counts["crop_sha256_verified"] == counts["rows"],
            "test_rows": 0,
            "frozen_markers": 0,
        },
        "policy": {
            **policy,
            "runtime_paths_audited": True,
            "labels_or_splits_changed": False,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "CC BY-NC research-only_non-deployable",
        },
        "decision": "Only this absolute-path, decode- and SHA-audited manifest may retry the isolated Stage80 color teacher; R1 failure evidence remains immutable.",
        "failures": [],
    }
    temporary_report = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary_report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, args.output_report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--expected-input-report-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = normalize(parse_args())
    print(json.dumps({
        "status": result["status"],
        "rows": result["output"]["rows"],
        "manifest_sha256": result["output"]["sha256"],
        "test_accessed": False,
        "frozen_video_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
