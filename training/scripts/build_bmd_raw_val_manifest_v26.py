#!/usr/bin/env python3
"""Append independent BMD-45 validation-source crops to the v23 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
    val_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(r"bmdraw_(\d+)_")
    for shard in args.shard:
        with shard.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_dataset") != "BMD-45-RAW":
                    continue
                path = row.get("image_path", "")
                if path in seen:
                    continue
                match = pattern.search(path)
                if not match:
                    continue
                seen.add(path)
                image_id = int(match.group(1))
                row["split"] = "validation" if image_id % 2 == 0 else "test"
                row["source_dataset"] = "BMD-45-RAW-VAL"
                row["source_group"] = "BMD-45-RAW-VAL"
                row["review_method"] = "deterministic source-category mapping; image-group holdout"
                val_rows.append(row)
    merged = base_rows + val_rows
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "schema_version": "bmd-raw-independent-val-manifest-v26",
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha(args.base_manifest),
        "shards": [{"path": str(p), "sha256": sha(p)} for p in args.shard],
        "rows_before": len(base_rows),
        "independent_raw_val_rows_added": len(val_rows),
        "rows_after": len(merged),
        "split_counts": dict(sorted(Counter(row["split"] for row in val_rows).items())),
        "type_counts": dict(sorted(Counter(row["body_type"] for row in val_rows).items())),
        "color_supervised_added": sum(row.get("color_supervised", "").lower() in {"true", "1", "yes"} for row in val_rows),
        "policy": "BMD-45 official validation images are image-group split into validation/test; no raw validation crop is used for training or threshold selection outside validation; no color pseudo-labels.",
        "frozen_video_used": False,
        "output_manifest": str(args.output_manifest),
    }
    report["output_manifest_sha256"] = sha(args.output_manifest)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
