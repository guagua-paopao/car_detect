#!/usr/bin/env python3
"""Merge sharded raw BMD crop manifests with the untouched base manifest."""

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
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    if not base_rows:
        raise SystemExit("base manifest is empty")
    fields = list(base_rows[0])
    new_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    shard_reports = []
    for path in args.shard:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        shard_new = [row for row in rows if row.get("source_dataset") == "BMD-45-RAW"]
        for row in shard_new:
            key = row.get("image_path", "")
            if key and key not in seen:
                seen.add(key)
                new_rows.append(row)
        shard_reports.append({"path": str(path), "sha256": sha(path), "rows": len(shard_new)})
    merged = base_rows + new_rows
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "schema_version": "bmd-raw-attribute-manifest-v23-merged",
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha(args.base_manifest),
        "shards": shard_reports,
        "rows_before": len(base_rows),
        "raw_crops_added": len(new_rows),
        "rows_after": len(merged),
        "raw_type_counts": dict(sorted(Counter(row.get("body_type", "") for row in new_rows).items())),
        "duplicate_raw_paths_removed": sum(report["rows"] for report in shard_reports) - len(new_rows),
        "policy": "Original base validation/test rows are preserved; only deterministic source-category type crops are appended; no raw color pseudo-labels.",
        "frozen_video_used": False,
    }
    report["output_manifest"] = str(args.output_manifest)
    report["output_manifest_sha256"] = sha(args.output_manifest)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
