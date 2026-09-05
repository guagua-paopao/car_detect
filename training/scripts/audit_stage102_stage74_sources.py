#!/usr/bin/env python3
"""Audit Stage74 train-only color-source provenance before Stage102 reuse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def source_key(row: dict[str, str]) -> str:
    return " | ".join(
        row.get(field, "").strip() or "<empty>"
        for field in (
            "stage74_joint_source",
            "stage71_origin",
            "source_dataset",
            "source_license",
            "formal_train_eligible",
            "license_train_eligible",
        )
    )


def core_eligible(row: dict[str, str]) -> bool:
    return (
        row.get("split") == "train"
        and row.get("review_status") == "approved"
        and truthy(row.get("formal_train_eligible"))
        and truthy(row.get("license_train_eligible"))
        and bool(row.get("source_license", "").strip())
        and truthy(row.get("color_supervised"))
        and row.get("color") not in {"", "unknown"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    actual_sha = sha256(manifest)
    if actual_sha.lower() != args.expected_sha256.lower():
        raise ValueError("Stage74 manifest SHA256 mismatch")

    counters: dict[str, Counter[str]] = {
        "eligible_stage74_joint_sources": Counter(),
        "eligible_stage71_cctv_sources": Counter(),
        "eligible_ua_sources": Counter(),
        "eligible_bmd_sources": Counter(),
        "eligible_openimages_sources": Counter(),
        "all_stage74_joint_sources": Counter(),
    }
    colors: dict[str, Counter[str]] = {key: Counter() for key in counters}
    rows_seen = 0
    train_rows = 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Stage74 manifest has no header")
        for row in reader:
            rows_seen += 1
            if row.get("split") != "train":
                continue
            train_rows += 1
            joint = row.get("stage74_joint_source", "").strip().lower()
            origin = row.get("stage71_origin", "").strip().lower()
            dataset = row.get("source_dataset", "").strip().lower()
            key = source_key(row)
            if joint:
                counters["all_stage74_joint_sources"][key] += 1
                colors["all_stage74_joint_sources"][row.get("color", "unknown")] += 1
            if not core_eligible(row):
                continue
            categories: list[str] = []
            if joint:
                categories.append("eligible_stage74_joint_sources")
            if origin == "cctv":
                categories.append("eligible_stage71_cctv_sources")
            if "ua" in joint or "ua" in dataset:
                categories.append("eligible_ua_sources")
            if "bmd" in joint or "bmd" in dataset:
                categories.append("eligible_bmd_sources")
            if "open-images" in dataset or "openimages" in dataset:
                categories.append("eligible_openimages_sources")
            for category in categories:
                counters[category][key] += 1
                colors[category][row.get("color", "unknown")] += 1

    report = {
        "schema_version": "stage102-stage74-source-provenance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_training_manifest_source_audit",
        "manifest": str(manifest),
        "manifest_sha256": actual_sha,
        "rows_seen": rows_seen,
        "train_rows": train_rows,
        "source_combinations": {
            name: dict(sorted(counter.items())) for name, counter in counters.items()
        },
        "color_counts": {
            name: dict(sorted(counter.items())) for name, counter in colors.items()
        },
        "policy": {
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
