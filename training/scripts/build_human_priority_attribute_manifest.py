#!/usr/bin/env python3
"""Disable pseudo body labels while retaining useful color supervision."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if "body_type_supervised" not in fieldnames:
        fieldnames.append("body_type_supervised")

    disabled = 0
    retained_body = Counter()
    retained_color = Counter()
    for row in rows:
        if (
            row.get("split") == "train"
            and row.get("review_status") == "approved"
            and row.get("annotation_source") == "calibrated_vlm"
        ):
            row["body_type_supervised"] = "false"
            disabled += 1
        if (
            row.get("split") == "train"
            and row.get("review_status") == "approved"
        ):
            if row.get("body_type_supervised", "true").lower() != "false":
                retained_body[row.get("body_type", "")] += 1
            if row.get("color_supervised", "true").lower() != "false":
                retained_color[row.get("color", "")] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: disabled pseudo body supervision={disabled}")
    print(f"retained train body={dict(retained_body)}")
    print(f"retained train color={dict(retained_color)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
