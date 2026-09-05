#!/usr/bin/env python3
"""Build contact sheets for human review of VLM/approved-label disagreements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from textwrap import shorten
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


FIELDS = ("body_type", "color", "crop_quality", "viewpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--items-per-sheet", type=int, default=24)
    parser.add_argument("--only-approved", action="store_true")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="optionally restrict the review batch to one split",
    )
    parser.add_argument(
        "--maximum-items",
        type=int,
        default=0,
        help="keep only the highest-priority N disagreements; 0 keeps all",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    return {str(value["image_path"]): value for value in values}


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
    manifest = args.manifest.resolve()
    root = manifest.parent
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictions = read_predictions(args.predictions)
    review: list[dict[str, Any]] = []
    for row in rows:
        if args.only_approved and row.get("review_status") != "approved":
            continue
        if args.split and row.get("split") != args.split:
            continue
        record = predictions.get(row["image_path"])
        prediction = record.get("prediction") if record else None
        if not isinstance(prediction, dict):
            continue
        disagreements = [
            field for field in FIELDS if row.get(field) != prediction.get(field)
        ]
        if not disagreements:
            continue
        item = dict(row)
        item["disagreements"] = ",".join(disagreements)
        item["disagreement_count"] = len(disagreements)
        for field in FIELDS:
            item[f"teacher_{field}"] = prediction.get(field, "")
            item[f"teacher_{field}_confidence"] = prediction.get(
                "confidence", {}
            ).get(field, "")
            item[f"reviewed_{field}"] = ""
        review.append(item)
    review.sort(
        key=lambda item: (
            -int(item["disagreement_count"]),
            item["split"],
            item["image_path"],
        )
    )
    if args.maximum_items > 0:
        review = review[: args.maximum_items]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(review[0]) if review else [
        "image_path",
        "disagreements",
        *[f"reviewed_{field}" for field in FIELDS],
    ]
    with (output_dir / "review_queue.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review)

    columns = 4
    rows_per_sheet = max(1, (args.items_per_sheet + columns - 1) // columns)
    cell_width = 360
    cell_height = 270
    image_size = (340, 180)
    font = ImageFont.load_default()
    for page_start in range(0, len(review), args.items_per_sheet):
        page = review[page_start : page_start + args.items_per_sheet]
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
            image_path = root / item["image_path"]
            with Image.open(image_path) as source:
                thumbnail = fit_image(source, image_size)
            canvas.paste(thumbnail, (x, y))
            absolute_index = page_start + offset
            old = f"{item.get('body_type')} | {item.get('color')} | {item.get('crop_quality')}"
            teacher = (
                f"{item.get('teacher_body_type')} | "
                f"{item.get('teacher_color')} | "
                f"{item.get('teacher_crop_quality')}"
            )
            caption = [
                f"#{absolute_index} {shorten(item['image_path'], width=48)}",
                f"old: {old}",
                f"vlm: {teacher}",
                f"diff: {item['disagreements']}",
            ]
            draw.multiline_text(
                (x, y + image_size[1] + 4),
                "\n".join(caption),
                fill="black",
                font=font,
                spacing=2,
            )
        page_number = page_start // args.items_per_sheet + 1
        canvas.save(output_dir / f"review_sheet_{page_number:03d}.jpg", quality=92)

    summary = {
        "manifest": str(manifest),
        "predictions": str(args.predictions.resolve()),
        "review_items": len(review),
        "sheets": (
            (len(review) + args.items_per_sheet - 1) // args.items_per_sheet
        ),
        "review_queue": str(output_dir / "review_queue.csv"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
