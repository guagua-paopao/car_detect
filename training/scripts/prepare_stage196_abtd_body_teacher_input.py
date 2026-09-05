#!/usr/bin/env python3
"""Prepare only Stage195 fail-closed eligible ABTD crops for body-teacher review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for value in (args.manifest, args.output):
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.output.exists():
        raise FileExistsError(args.output)

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "stage195_train_eligible" not in fields or "crop_path" not in fields:
        raise RuntimeError("not a Stage195 crop manifest")

    selected = []
    for row in rows:
        if row.get("split") != "train" or not truthy(row.get("stage195_train_eligible")):
            continue
        if not truthy(row.get("body_type_supervised")) or row.get("body_type") not in {"bus", "truck"}:
            continue
        crop = Path(str(row.get("crop_path") or ""))
        if not crop.is_absolute() or not crop.is_file():
            raise RuntimeError(f"missing absolute crop: {crop}")
        prepared = dict(row)
        prepared["image_path"] = str(crop)
        prepared["formal_train_eligible"] = "true"
        selected.append(prepared)
    if len(selected) < 300 or {row["body_type"] for row in selected} != {"bus", "truck"}:
        raise RuntimeError("insufficient two-class Stage195 teacher input")

    output_fields = fields + [field for field in ("image_path", "formal_train_eligible") if field not in fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"prepared_rows={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
