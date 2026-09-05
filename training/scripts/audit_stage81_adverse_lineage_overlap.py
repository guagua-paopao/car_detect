#!/usr/bin/env python3
"""Audit whether licensed adverse rows are already present in a successor manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRUTHY = {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str:
    for field in ("sha256", "image_sha256", "image_path"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def build_report(source: Path, successor: Path) -> dict[str, object]:
    source_rows = read_rows(source)
    successor_rows = read_rows(successor)
    source_keys = {row_key(row) for row in source_rows if row_key(row)}
    successor_keys = {row_key(row) for row in successor_rows if row_key(row)}

    adverse_rows = [row for row in source_rows if truthy(row.get("stage66_licensed_adverse"))]
    adverse_keys = {row_key(row) for row in adverse_rows if row_key(row)}
    missing = [row for row in adverse_rows if row_key(row) not in successor_keys]
    missing_body_supervised = [row for row in missing if truthy(row.get("body_type_supervised"))]
    missing_color_supervised = [row for row in missing if truthy(row.get("color_supervised"))]

    adverse_overlap = len(adverse_keys & successor_keys)
    overlap_fraction = adverse_overlap / len(adverse_keys) if adverse_keys else 0.0
    failures: list[str] = []
    if not source_keys or not successor_keys:
        failures.append("manifest contains no usable row keys")
    if overlap_fraction < 0.99:
        failures.append(f"adverse overlap below 0.99: {overlap_fraction:.6f}")
    if missing_body_supervised:
        failures.append(f"missing supervised body adverse rows: {len(missing_body_supervised)}")
    if missing_color_supervised:
        failures.append(f"missing supervised color adverse rows: {len(missing_color_supervised)}")

    return {
        "schema_version": "stage81-adverse-lineage-overlap-audit-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "manifest": str(source.resolve()),
            "sha256": sha256(source),
            "rows": len(source_rows),
            "unique_keys": len(source_keys),
        },
        "successor": {
            "manifest": str(successor.resolve()),
            "sha256": sha256(successor),
            "rows": len(successor_rows),
            "unique_keys": len(successor_keys),
        },
        "overlap": {
            "all_source_unique_keys": len(source_keys & successor_keys),
            "all_source_fraction": len(source_keys & successor_keys) / len(source_keys) if source_keys else 0.0,
            "licensed_adverse_rows": len(adverse_rows),
            "licensed_adverse_unique_keys": len(adverse_keys),
            "licensed_adverse_overlap_unique_keys": adverse_overlap,
            "licensed_adverse_overlap_fraction": overlap_fraction,
            "missing_adverse_rows": len(missing),
            "missing_body_supervised_rows": len(missing_body_supervised),
            "missing_color_supervised_rows": len(missing_color_supervised),
            "missing_body_labels": dict(Counter(row.get("body_type", "") for row in missing_body_supervised)),
            "missing_color_labels": dict(Counter(row.get("color", "") for row in missing_color_supervised)),
            "missing_examples": [
                {
                    key: row.get(key, "")
                    for key in (
                        "image_path", "sha256", "split", "body_type", "body_type_supervised",
                        "color", "color_supervised", "source_dataset", "lighting", "vehicle_size",
                        "occluded", "truncated",
                    )
                }
                for row in missing[:20]
            ],
        },
        "decision": {
            "reimport_stage66": False if not failures else None,
            "reason": (
                "All licensed-adverse rows carrying supervised truth are already represented; "
                "reimporting Stage66 would add duplicates rather than new supervision."
                if not failures
                else "Resolve the recorded failures before deciding whether any missing supervised rows are eligible."
            ),
            "remaining_gap": "new licensed real night/occluded samples with reliable truth",
        },
        "policy": {
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--successor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.source, args.successor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
