#!/usr/bin/env python3
"""Build a balanced visual-review pack for replacing rejected attribute crops."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from textwrap import shorten

from PIL import Image, ImageDraw, ImageFont, ImageOps


BODY_TYPES = (
    "sedan",
    "suv",
    "mpv",
    "van",
    "pickup",
    "bus",
    "light_truck",
    "heavy_truck",
    "other",
)
COLORS = (
    "black",
    "white",
    "silver_gray",
    "red",
    "blue",
    "green",
    "yellow_orange",
    "brown_beige",
    "other",
)
FIELDS = ("body_type", "color", "crop_quality", "viewpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-queue", type=Path, required=True)
    parser.add_argument(
        "--exclude-reviews",
        type=Path,
        help="Optional completed review CSV whose image paths must be excluded.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-items", type=int, default=600)
    parser.add_argument("--items-per-sheet", type=int, default=24)
    return parser.parse_args()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return canvas


def main() -> int:
    args = parse_args()
    queue_path = args.manual_queue.resolve()
    root = queue_path.parent
    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    excluded: set[str] = set()
    if args.exclude_reviews:
        with args.exclude_reviews.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            excluded = {row["image_path"] for row in csv.DictReader(handle)}

    candidates = []
    for source_row in rows:
        candidate_body = (
            source_row.get("suggested_body_type") or source_row.get("body_type")
        )
        candidate_color = (
            source_row.get("suggested_color") or source_row.get("color")
        )
        candidate_quality = (
            source_row.get("suggested_crop_quality")
            or source_row.get("crop_quality")
        )
        if not (
            source_row.get("split") == args.split
            and source_row["image_path"] not in excluded
            and candidate_quality == "good"
            and candidate_body in BODY_TYPES
            and candidate_color in COLORS
            and all(
                source_row.get(flag) == "false"
                for flag in ("blur", "occluded", "truncated", "night")
            )
        ):
            continue
        row = dict(source_row)
        row["_candidate_body_type"] = candidate_body
        row["_candidate_color"] = candidate_color
        row["_candidate_crop_quality"] = candidate_quality
        candidates.append(row)
    candidates.sort(key=lambda row: row["image_path"])

    # Keep one crop per source frame so the replacement set is visually diverse.
    unique_candidates = []
    seen_sources: set[str] = set()
    for row in candidates:
        source = row.get("source_frame_id") or row["image_path"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        unique_candidates.append(row)

    buckets: dict[tuple[str, str], deque[dict[str, str]]] = defaultdict(deque)
    for row in unique_candidates:
        buckets[
            (row["_candidate_body_type"], row["_candidate_color"])
        ].append(row)

    selected: list[dict[str, str]] = []
    selected_colors: Counter[str] = Counter()
    selected_bodies: Counter[str] = Counter()
    while len(selected) < args.maximum_items:
        progressed = False
        for body in sorted(BODY_TYPES, key=lambda value: (selected_bodies[value], value)):
            available_colors = [
                color for color in COLORS if buckets[(body, color)]
            ]
            if not available_colors:
                continue
            color = min(
                available_colors,
                key=lambda value: (selected_colors[value], value),
            )
            row = buckets[(body, color)].popleft()
            item = dict(row)
            item["teacher_body_type"] = row["_candidate_body_type"]
            item["teacher_color"] = row["_candidate_color"]
            item["teacher_crop_quality"] = row["_candidate_crop_quality"]
            item["teacher_viewpoint"] = row.get("viewpoint", "unknown")
            for private_field in (
                "_candidate_body_type",
                "_candidate_color",
                "_candidate_crop_quality",
            ):
                item.pop(private_field, None)
            for field in FIELDS:
                item[f"reviewed_{field}"] = ""
            selected.append(item)
            selected_bodies[body] += 1
            selected_colors[color] += 1
            progressed = True
            if len(selected) >= args.maximum_items:
                break
        if not progressed:
            break

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0]) if selected else [
        "image_path",
        *[f"reviewed_{field}" for field in FIELDS],
    ]
    queue_output = output_dir / "review_queue.csv"
    with queue_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    columns = 4
    rows_per_sheet = (
        args.items_per_sheet + columns - 1
    ) // columns
    cell_width = 360
    cell_height = 270
    image_size = (340, 180)
    font = ImageFont.load_default()
    for page_start in range(0, len(selected), args.items_per_sheet):
        page = selected[page_start : page_start + args.items_per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * cell_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for offset, item in enumerate(page):
            column = offset % columns
            row_index = offset // columns
            x = column * cell_width + 10
            y = row_index * cell_height + 8
            with Image.open(root / item["image_path"]) as source:
                thumbnail = fit_image(source, image_size)
            canvas.paste(thumbnail, (x, y))
            absolute_index = page_start + offset
            caption = [
                f"#{absolute_index} {shorten(item['image_path'], width=48)}",
                (
                    f"candidate: {item['teacher_body_type']} | "
                    f"{item['teacher_color']} | good"
                ),
                f"old: {item.get('body_type')} | {item.get('color')}",
            ]
            draw.multiline_text(
                (x, y + image_size[1] + 4),
                "\n".join(caption),
                fill="black",
                font=font,
                spacing=2,
            )
        page_number = page_start // args.items_per_sheet + 1
        canvas.save(
            output_dir / f"review_sheet_{page_number:03d}.jpg",
            quality=92,
        )

    summary = {
        "eligible_before_source_dedup": len(candidates),
        "eligible_after_source_dedup": len(unique_candidates),
        "selected": len(selected),
        "split": args.split,
        "body_distribution": dict(selected_bodies),
        "color_distribution": dict(selected_colors),
        "sheets": (
            len(selected) + args.items_per_sheet - 1
        ) // args.items_per_sheet,
        "review_queue": str(queue_output),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
