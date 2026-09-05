#!/usr/bin/env python3
"""Create deterministic contact sheets for visual attribute dataset review."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root for image_path values; defaults to the manifest parent.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", choices=("body_type", "color"), required=True)
    parser.add_argument("--per-group", type=int, default=25)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else manifest.parent
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("review_status") == "approved"]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        supervised = row.get(f"{args.field}_supervised", "true").strip().lower()
        if supervised not in {"true", "1", "yes"}:
            continue
        groups[(row["split"], row[args.field])].append(row)
    font = ImageFont.load_default()
    tile_width, tile_height, caption_height = 224, 160, 42
    written = 0
    for (split, label), candidates in sorted(groups.items()):
        random.Random(f"{args.seed}:{split}:{label}").shuffle(candidates)
        chosen = candidates[: args.per_group]
        row_count = (len(chosen) + args.columns - 1) // args.columns
        sheet = Image.new(
            "RGB",
            (args.columns * tile_width, row_count * (tile_height + caption_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chosen):
            path = dataset_root / row["image_path"]
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = (index % args.columns) * tile_width
            y = (index // args.columns) * (tile_height + caption_height)
            paste_x = x + (tile_width - image.width) // 2
            paste_y = y + (tile_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
            caption = (
                f"{label} | {row.get('crop_quality', '')}\n"
                f"{Path(row['image_path']).name[:28]}"
            )
            draw.multiline_text((x + 3, y + tile_height + 2), caption, fill="black", font=font, spacing=2)
        output = output_dir / f"{args.field}_{split}_{label}.jpg"
        sheet.save(output, quality=92)
        written += 1
    print(f"PASS: wrote {written} contact sheets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
