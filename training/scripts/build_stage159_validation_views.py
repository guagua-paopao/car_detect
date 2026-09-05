#!/usr/bin/env python3
"""Seal validation-only body and color manifests for Stage159 calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48", "stage148", "stage155")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def write(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seal_view(
    rows: list[dict[str, str]],
    head: str,
    dataset_root: Path,
    safety_roots: tuple[Path, ...] | None = None,
) -> tuple[list[dict[str, str]], dict]:
    allowed_roots = tuple(root.resolve() for root in (safety_roots or (dataset_root,)))
    supervised_key = f"{head}_supervised" if head == "color" else "body_type_supervised"
    label_key = "color" if head == "color" else "body_type"
    selected = [dict(row) for row in rows if row.get("split") == "validation" and truthy(row.get(supervised_key))]
    if not selected:
        raise RuntimeError(f"no supervised {head} validation rows")
    groups = set()
    for line, row in enumerate(selected, start=2):
        searchable = " ".join(str(value).lower() for value in row.values())
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"forbidden marker in {head} row {line}")
        if row.get("review_status") != "approved":
            raise RuntimeError(f"unapproved {head} validation row {line}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (dataset_root / raw).resolve()
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"validation path escapes allowed safety roots: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["image_path"] = str(resolved)
        row["split"] = "validation"
        group = row.get("stage157_group") or row.get("track_group") or row.get("track_key") or row.get("track_id") or row.get("source_group") or row.get("source_frame_id")
        if not group:
            raise RuntimeError(f"missing validation group in {head} row {line}")
        row["stage159_validation_group"] = group
        groups.add(group)
    summary = {
        "rows": len(selected),
        "groups": len(groups),
        "label_counts": dict(sorted(Counter(row.get(label_key, "") for row in selected).items())),
        "source_counts": dict(sorted(Counter(row.get("source_dataset", "") for row in selected).items())),
        "all_paths_absolute_readable": True,
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-manifest", type=Path, required=True)
    parser.add_argument("--expected-body-manifest-sha256", required=True)
    parser.add_argument("--color-manifest", type=Path, required=True)
    parser.add_argument("--expected-color-manifest-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources-root", type=Path, required=True)
    parser.add_argument("--output-body", type=Path, required=True)
    parser.add_argument("--output-color", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.body_manifest, args.color_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_body, args.output_color, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage159 evidence: {path}")
    body_sha = sha256_file(args.body_manifest)
    color_sha = sha256_file(args.color_manifest)
    if body_sha != args.expected_body_manifest_sha256.lower():
        raise RuntimeError("body manifest SHA256 mismatch")
    if color_sha != args.expected_color_manifest_sha256.lower():
        raise RuntimeError("color manifest SHA256 mismatch")
    safety_roots = (args.dataset_root.parent, args.sources_root)
    body_rows, body_summary = seal_view(read(args.body_manifest), "body_type", args.dataset_root, safety_roots)
    color_rows, color_summary = seal_view(read(args.color_manifest), "color", args.dataset_root, safety_roots)
    if body_summary["rows"] < 10000:
        raise RuntimeError("body validation support below 10000")
    if color_summary["rows"] < 4500:
        raise RuntimeError("color-supervised validation support below 4500")
    for color in ("black", "white", "gray", "silver", "red", "blue", "green", "yellow", "brown", "other"):
        if color_summary["label_counts"].get(color, 0) < 200:
            raise RuntimeError(f"color validation support below 200 for {color}")
    args.output_body.parent.mkdir(parents=True, exist_ok=True)
    write(args.output_body, body_rows)
    write(args.output_color, color_rows)
    report = {
        "schema_version": "stage159-validation-only-views-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_validation_only",
        "inputs": {
            "body_manifest": str(args.body_manifest.resolve()),
            "body_manifest_sha256": body_sha,
            "color_manifest": str(args.color_manifest.resolve()),
            "color_manifest_sha256": color_sha,
            "allowed_image_roots": [str(path.resolve()) for path in safety_roots],
        },
        "output": {
            "body_manifest": str(args.output_body.resolve()),
            "body_manifest_sha256": sha256_file(args.output_body),
            "body": body_summary,
            "color_manifest": str(args.output_color.resolve()),
            "color_manifest_sha256": sha256_file(args.output_color),
            "color": color_summary,
        },
        "policy": {
            "validation_only": True,
            "threshold_and_temperature_selection_allowed": True,
            "model_training_allowed": False,
            "test_accessed": False,
            "stage148_or_stage155_reused": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "body": body_summary, "color": color_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
