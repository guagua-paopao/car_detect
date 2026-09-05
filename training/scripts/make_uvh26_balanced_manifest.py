#!/usr/bin/env python3
"""Limit UVH-26's type-only supplement to protect the color head."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-uvh-rows", type=int, default=3000)
    args = ap.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f); rows = list(reader); fields = reader.fieldnames or []
    base = [r for r in rows if (r.get("source_dataset") or "") != "UVH-26"]
    uvh = [r for r in rows if (r.get("source_dataset") or "") == "UVH-26"]
    groups = defaultdict(list)
    for r in uvh: groups[r.get("body_type", "unknown")].append(r)
    chosen = []
    # Round-robin by mapped class keeps minority van/MPV represented.
    keys = sorted(groups)
    while len(chosen) < min(args.max_uvh_rows, len(uvh)):
        progressed = False
        for k in keys:
            if groups[k] and len(chosen) < args.max_uvh_rows:
                chosen.append(groups[k].pop(0)); progressed = True
        if not progressed: break
    out_rows = base + chosen
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    report = {"schema_version":"uvh26-color-safe-manifest-v1", "input_rows":len(rows), "base_rows":len(base), "uvh_rows_available":len(uvh), "uvh_rows_selected":len(chosen), "output_rows":len(out_rows), "frozen_video_used":False, "purpose":"protect_color_head_from_type_only_domain_shift"}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
