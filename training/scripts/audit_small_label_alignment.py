#!/usr/bin/env python3
"""Audit small-target supervision and source/label alignment without touching frozen video."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    def is_small(row):
        return str(row.get("small_target", "")).lower() in {"1", "true", "yes"} or str(row.get("vehicle_size", "")).lower() == "small"
    report = {"schema_version": "small-label-alignment-audit-v1", "manifest": str(args.manifest), "rows": len(rows), "frozen_video_used": False, "splits": {}}
    for split in sorted({r.get("split", "") for r in rows}):
        subset = [r for r in rows if r.get("split") == split and r.get("body_type") not in {"unknown", ""}]
        small = [r for r in subset if is_small(r)]
        by_source = defaultdict(Counter)
        for row in small:
            by_source[row.get("source_dataset", "unknown")][row.get("body_type", "unknown")] += 1
        report["splits"][split] = {"labeled_rows": len(subset), "small_labeled_rows": len(small), "small_fraction": len(small) / len(subset) if subset else 0.0, "small_by_source_class": {src: dict(counts) for src, counts in sorted(by_source.items())}}
    report["findings"] = [
        "small_target and vehicle_size=small are treated as equivalent for sampling audits",
        "source_dataset and body_type counts are reported to detect domain/class imbalance",
        "unknown or unlabeled rows are excluded from labeled small-target counts",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
