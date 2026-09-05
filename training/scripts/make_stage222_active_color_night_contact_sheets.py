#!/usr/bin/env python3
"""Render the effective active-train explicit-night color rows for agent audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


TRUE_VALUES = {"1", "true", "yes", "y"}


def is_true(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if str(row.get("split") or "").strip().lower() == "train"
        and is_true(row.get("color_supervised"))
        and is_true(row.get("stage177_dedup_eligible"))
        and is_true(row.get("stage177_effective_representative"))
        and str(row.get("stage177_scene_label") or "").strip().lower() == "night"
    ]
    selected.sort(key=lambda row: (
        str(row.get("source_dataset") or ""),
        str(row.get("color") or ""),
        str(row.get("stage177_group_key") or ""),
        str(row.get("sha256") or row.get("stage177_pixel_sha256") or ""),
    ))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=144)
    parser.add_argument("--per-sheet", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(f"manifest must be a regular non-symlink file: {manifest}")
    actual_sha = sha256_file(manifest)
    if actual_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_sha}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.per_sheet <= 0 or args.columns <= 0:
        raise ValueError("per-sheet and columns must be positive")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = select_rows(list(csv.DictReader(handle)))
    if len(rows) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} effective explicit-night rows, got {len(rows)}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    tile_w, tile_h = 300, 250
    sheet_rows = math.ceil(args.per_sheet / args.columns)
    sheet_records: list[dict[str, Any]] = []
    row_index: list[dict[str, Any]] = []

    for sheet_index, start in enumerate(range(0, len(rows), args.per_sheet), 1):
        page_rows = rows[start:start + args.per_sheet]
        sheet = Image.new("RGB", (args.columns * tile_w, sheet_rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for page_offset, row in enumerate(page_rows):
            global_index = start + page_offset + 1
            path = Path(str(row.get("image_path") or ""))
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise RuntimeError(f"missing or unsafe active train image: {path}")
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 76))
            column, page_row = page_offset % args.columns, page_offset // args.columns
            x0, y0 = column * tile_w, page_row * tile_h
            x = x0 + (tile_w - image.width) // 2
            y = y0 + 72 + (tile_h - 76 - image.height) // 2
            sheet.paste(image, (x, y))
            source = str(row.get("source_dataset") or "?")
            color = str(row.get("color") or "?")
            mean = float(row.get("stage177_mean_luma") or 0.0)
            draw.text((x0 + 4, y0 + 4), f"#{global_index:03d} {source[:23]}", fill="black")
            draw.text((x0 + 4, y0 + 22), f"night {color} mean={mean:.1f}", fill="black")
            draw.text((x0 + 4, y0 + 40), str(row.get("stage177_group_key") or "")[:35], fill="black")
            draw.text((x0 + 4, y0 + 56), str(row.get("sha256") or row.get("stage177_pixel_sha256") or "")[:20], fill="black")
            row_index.append({
                "index": global_index,
                "sheet": sheet_index,
                "image_path": str(path),
                "sha256": row.get("sha256") or row.get("stage177_pixel_sha256"),
                "source_dataset": source,
                "source_license": row.get("source_license"),
                "color": color,
                "group_key": row.get("stage177_group_key"),
                "scene_origin": row.get("stage177_scene_origin"),
                "mean_luma": mean,
                "lowlight_score": float(row.get("stage177_lowlight_score") or 0.0),
            })
        output = args.output_root / f"stage222-active-night-contact-sheet-{sheet_index:02d}.png"
        sheet.save(output, format="PNG", optimize=True)
        sheet_records.append({
            "sheet": sheet_index,
            "first_index": start + 1,
            "last_index": start + len(page_rows),
            "path": str(output),
            "sha256": sha256_file(output),
        })

    report = {
        "schema_version": "stage222-active-color-night-contact-sheets-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_agent_visual_review_material_ready",
        "input_manifest": str(manifest),
        "input_manifest_sha256": actual_sha,
        "selection": {
            "split": "train",
            "color_supervised": True,
            "stage177_dedup_eligible": True,
            "stage177_effective_representative": True,
            "stage177_scene_label": "night",
        },
        "rows": len(rows),
        "source_counts": dict(sorted(Counter(r.get("source_dataset") or "unknown" for r in rows).items())),
        "color_counts": dict(sorted(Counter(r.get("color") or "unknown" for r in rows).items())),
        "sheets": sheet_records,
        "row_index": row_index,
        "policy": {
            "train_only": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "labels_modified": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage222-active-night-contact-sheet-index.json"
    atomic_json(report_path, report)
    report_path.with_suffix(".json.sha256").write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "rows": len(rows),
        "source_counts": report["source_counts"],
        "sheet_count": len(sheet_records),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
