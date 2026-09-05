#!/usr/bin/env python3
"""Mask non-small train rows to train a body-type small-target specialist."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    train_small = 0
    train_masked = 0
    for row in rows:
        if row.get("split") != "train":
            continue
        is_small = row.get("vehicle_size") == "small" or truth(row.get("small_target", ""))
        if is_small and truth(row.get("body_type_supervised", "true")) and row.get("body_type") not in {"", "unknown"}:
            # Keep only the type target for the specialist.
            row["color_supervised"] = "false"
            train_small += 1
        else:
            row["body_type_supervised"] = "false"
            row["color_supervised"] = "false"
            row["body_type"] = "unknown"
            row["color"] = "unknown"
            train_masked += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "small-specialist-manifest-v1",
        "input_rows": len(rows),
        "output_rows": len(rows),
        "train_small_body_rows": train_small,
        "train_masked_rows": train_masked,
        "validation_test_unchanged": True,
        "color_supervision_train_disabled": True,
        "frozen_video_used": False,
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
