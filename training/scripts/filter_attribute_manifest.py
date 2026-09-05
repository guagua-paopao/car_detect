#!/usr/bin/env python3
"""Write a non-destructive subset of an attribute CSV by review status."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-status", required=True)
    parser.add_argument("--split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        fieldnames = list(reader.fieldnames)
        rows = [
            row
            for row in reader
            if row.get("review_status") == args.review_status
            and (args.split is None or row.get("split") == args.split)
        ]
    if not rows:
        raise RuntimeError(
            f"no rows have review_status={args.review_status!r}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS: wrote {len(rows)} rows with review_status="
        f"{args.review_status!r}, split={args.split!r} to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
