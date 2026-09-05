#!/usr/bin/env python3
"""Combine the sealed Stage157 train and validation views for the trainer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.split_report, args.train_manifest, args.validation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
    if split_report.get("status") != "pass_research_only":
        raise RuntimeError("Stage157 split did not pass")
    expected_train = split_report.get("output", {}).get("training_manifest_sha256")
    expected_validation = split_report.get("output", {}).get("validation_manifest_sha256")
    if sha256_file(args.train_manifest) != expected_train:
        raise RuntimeError("training manifest SHA256 mismatch")
    if sha256_file(args.validation_manifest) != expected_validation:
        raise RuntimeError("validation manifest SHA256 mismatch")
    train = read(args.train_manifest)
    validation = read(args.validation_manifest)
    if not train or not validation:
        raise RuntimeError("empty split")
    if any(row.get("split") != "train" for row in train):
        raise RuntimeError("non-train row in training manifest")
    if any(row.get("split") != "validation" for row in validation):
        raise RuntimeError("non-validation row in validation manifest")
    rows = train + validation
    for row in rows:
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found")
        if row.get("split") == "test":
            raise RuntimeError("test row found")
        raw_path = Path(row.get("image_path", ""))
        resolved = raw_path.resolve() if raw_path.is_absolute() else (args.dataset_root / raw_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if not raw_path.is_absolute():
            row["stage157_original_image_path"] = row.get("image_path", "")
        row["image_path"] = str(resolved)
    train_groups = {row.get("stage157_group", "") for row in train}
    validation_groups = {row.get("stage157_group", "") for row in validation}
    if "" in train_groups or "" in validation_groups or train_groups & validation_groups:
        raise RuntimeError("group contract failed")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "stage157-combined-color-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_only",
        "inputs": {
            "split_report": str(args.split_report.resolve()),
            "split_report_sha256": sha256_file(args.split_report),
            "training_manifest_sha256": expected_train,
            "validation_manifest_sha256": expected_validation,
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256_file(args.output_manifest),
            "rows": len(rows),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_color_counts": dict(sorted(Counter(row.get("color", "") for row in train if row.get("color_supervised") == "true").items())),
            "validation_color_counts": dict(sorted(Counter(row.get("color", "") for row in validation if row.get("color_supervised") == "true").items())),
        },
        "integrity": {"train_validation_group_overlap": len(train_groups & validation_groups)},
        "policy": {
            "all_image_paths_absolute_and_readable": True,
            "required_initialization": "fresh ImageNet weights outside Stage80/103/150 color lineage",
            "test_rows_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "train": len(train), "validation": len(validation)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
