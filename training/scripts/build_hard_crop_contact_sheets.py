#!/usr/bin/env python3
"""Create deterministic contact sheets for unlabeled Open Images hard crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def chosen(rows: list[dict], count: int, salt: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['image_path']}".encode("utf-8")
        ).hexdigest(),
    )[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--field", choices=("official_vehicle_class", "body_type", "color"),
        default="official_vehicle_class",
    )
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--consensus-field", default="")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite contact-sheet evidence")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        consensus_field = args.consensus_field or (
            "color_teacher_consensus" if args.field == "color" else "teacher_consensus"
        )
        if args.accepted_only and row.get(consensus_field) != "accepted":
            continue
        label = row.get(args.field, "unknown") or "unknown"
        groups[f"label_{args.field}_{label}"].append(row)
        if row.get("vehicle_size") == "small":
            groups["condition_small"].append(row)
        if row.get("occluded", "").lower() == "true":
            groups["condition_occluded"].append(row)
        if row.get("truncated", "").lower() == "true":
            groups["condition_truncated"].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    tile_width, tile_height, caption_height = 240, 170, 48
    outputs = []
    for group, candidates in sorted(groups.items()):
        sample = chosen(candidates, args.per_group, group)
        row_count = (len(sample) + args.columns - 1) // args.columns
        sheet = Image.new(
            "RGB", (args.columns * tile_width, row_count * (tile_height + caption_height)), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(sample):
            path = (args.dataset_root / row["image_path"]).resolve()
            path.relative_to(args.dataset_root.resolve())
            with Image.open(path) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_width, tile_height))
            x = (index % args.columns) * tile_width
            y = (index // args.columns) * (tile_height + caption_height)
            sheet.paste(
                image,
                (x + (tile_width - image.width) // 2, y + (tile_height - image.height) // 2),
            )
            tags = ",".join(
                key for key, present in (
                    ("small", row.get("vehicle_size") == "small"),
                    ("occ", row.get("occluded", "").lower() == "true"),
                    ("trunc", row.get("truncated", "").lower() == "true"),
                ) if present
            )
            caption = f"{row.get(args.field, 'unknown')} | {tags}\n{Path(row['image_path']).name[:34]}"
            draw.multiline_text((x + 3, y + tile_height + 2), caption, fill="black", font=font)
        output = args.output_dir / f"{group}.jpg"
        sheet.save(output, quality=92)
        outputs.append(output)
    print("\n".join(str(path.resolve()) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
