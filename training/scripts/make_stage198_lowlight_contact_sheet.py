#!/usr/bin/env python3
"""Render every accepted Stage198 low-light proxy for agent visual audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.overlay.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("stage198_train_eligible") == "true"]
    if not rows:
        raise RuntimeError("no accepted Stage198 rows")
    rows.sort(key=lambda row: (row["source_dataset"], row["stage177_scene_group_key"], row["image_path"]))
    columns, tile_w, tile_h = 5, 250, 205
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, sheet_rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        path = Path(row["image_path"])
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"missing accepted image: {path}")
        with Image.open(path) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 48))
        column, sheet_row = index % columns, index // columns
        x0, y0 = column * tile_w, sheet_row * tile_h
        x = x0 + (tile_w - image.width) // 2
        y = y0 + 42 + (tile_h - 46 - image.height) // 2
        sheet.paste(image, (x, y))
        probability = float(row["stage198_min_probability"])
        draw.text((x0 + 4, y0 + 4), f"{index + 1:02d} {row['source_dataset']} p={probability:.3f}", fill="black")
        draw.text((x0 + 4, y0 + 20), path.name[:34], fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(f"rendered={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
