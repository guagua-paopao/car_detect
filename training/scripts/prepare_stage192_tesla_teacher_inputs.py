#!/usr/bin/env python3
"""Prepare isolated body/color inputs for the Stage192 Tesla teacher audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def write_view(rows: list[dict[str, str]], fields: list[str], output: Path, head: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = fields + [name for name in ("image_path", "formal_train_eligible") if name not in fields]
    prepared: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        eligible = truthy(row.get("stage191_train_eligible")) and bool(row.get("crop_path"))
        row["image_path"] = row.get("crop_path", "")
        row["formal_train_eligible"] = str(eligible).lower()
        if head == "body":
            row["body_type_supervised"] = str(eligible and truthy(row.get("body_type_supervised"))).lower()
            row["color_supervised"] = "false"
        else:
            row["color_supervised"] = str(eligible and truthy(row.get("color_supervised"))).lower()
            row["body_type_supervised"] = "false"
        prepared.append(row)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prepared)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--body-output", type=Path, required=True)
    parser.add_argument("--color-output", type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 3026 or "stage191_train_eligible" not in fields or "crop_path" not in fields:
        raise RuntimeError("unexpected Stage191 manifest contract")
    write_view(rows, fields, args.body_output, "body")
    write_view(rows, fields, args.color_output, "color")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
