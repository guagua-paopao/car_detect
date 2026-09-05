#!/usr/bin/env python3
"""Join Stage198 accepted proxies to Stage177 photometric evidence without relabeling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "image_path", "source_dataset", "video_id", "track_group",
    "stage177_lowlight_score", "stage177_mean_luma", "stage177_median_luma",
    "stage177_border_mean_luma", "stage177_dark_fraction", "stage177_highlight_fraction",
    "stage177_contrast", "stage177_frame_lowlight_proxy", "stage177_frame_night_candidate",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.overlay.open("r", encoding="utf-8-sig", newline="") as handle:
        accepted = {
            row["image_path"]: row for row in csv.DictReader(handle)
            if row.get("stage198_train_eligible") == "true"
        }
    joined = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_path") not in accepted:
                continue
            stage198 = accepted[row["image_path"]]
            joined.append({
                **{field: row.get(field, "") for field in FIELDS},
                "stage198_original_probability": stage198["stage198_original_probability"],
                "stage198_hflip_probability": stage198["stage198_hflip_probability"],
                "stage198_min_probability": stage198["stage198_min_probability"],
            })
    if len(joined) != len(accepted):
        raise RuntimeError(f"join mismatch accepted={len(accepted)} joined={len(joined)}")
    joined.sort(key=lambda row: float(row["stage198_min_probability"]), reverse=True)
    report = {"status": "inspection_only_no_relabeling", "rows": len(joined), "candidates": joined}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
