#!/usr/bin/env python3
"""Build a train-only BMD-45 body-type manifest while preserving independent eval rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty input manifest")
    fieldnames = list(rows[0])
    changed = 0
    kept_train = 0
    excluded_train = 0
    counts = Counter()
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        source = str(row.get("source_dataset", "")).strip().lower()
        body = str(row.get("body_type", "")).strip().lower()
        supervised = str(row.get("body_type_supervised", "true")).strip().lower() not in {"false", "0", "no"}
        eligible = source == "bmd-45" and supervised and body not in {"", "unknown"}
        if split == "train":
            if eligible:
                kept_train += 1
                counts[body] += 1
            else:
                excluded_train += 1
                for key in ("body_type", "body_type_supervised"):
                    if key in row:
                        row[key] = "unknown" if key == "body_type" else "false"
                        changed += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "bmd-type-manifest-v22",
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "rows": len(rows),
        "train_bmd_body_supervised": kept_train,
        "train_rows_excluded_from_body_supervision": excluded_train,
        "body_type_counts_train": dict(sorted(counts.items())),
        "non_train_rows_preserved": True,
        "frozen_video_used": False,
        "policy": "Only BMD-45 train rows with non-unknown body labels remain body-supervised; validation/test rows are unchanged for independent evaluation.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
