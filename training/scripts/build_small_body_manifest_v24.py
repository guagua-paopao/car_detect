#!/usr/bin/env python3
"""Keep only small-target body supervision for an isolated specialist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty manifest")
    fields = list(rows[0])
    kept = Counter()
    changed = 0
    for row in rows:
        if row.get("split") == "train":
            size = str(row.get("vehicle_size", "")).strip().lower()
            body = str(row.get("body_type", "")).strip().lower()
            supervised = str(row.get("body_type_supervised", "true")).strip().lower() not in {"false", "0", "no"}
            if size == "small" and supervised and body not in {"", "unknown"}:
                kept[body] += 1
            else:
                if "body_type" in row:
                    row["body_type"] = "unknown"
                if "body_type_supervised" in row:
                    row["body_type_supervised"] = "false"
                changed += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "small-body-manifest-v24",
        "input": str(args.input),
        "input_sha256": sha(args.input),
        "output": str(args.output),
        "output_sha256": sha(args.output),
        "rows": len(rows),
        "small_train_body_supervised": sum(kept.values()),
        "small_train_body_counts": dict(sorted(kept.items())),
        "train_rows_body_supervision_disabled": changed,
        "validation_test_preserved": True,
        "frozen_video_used": False,
        "policy": "Only explicitly small train rows supervise body type; validation/test remain untouched for independent specialist evaluation.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
