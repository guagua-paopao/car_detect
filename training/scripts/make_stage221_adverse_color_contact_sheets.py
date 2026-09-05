#!/usr/bin/env python3
"""Render Stage220 train-only candidates into paged sheets for agent review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no Stage220 rows")
    if any(str(row.get("split") or "").lower() != "train" for row in rows):
        raise RuntimeError("non-train row in Stage220 contact-sheet input")
    rows.sort(key=lambda row: (row.get("stage220_scene_claim", ""), row.get("color", ""), row.get("sha256", "")))

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
                raise RuntimeError(f"missing or unsafe Stage220 train image: {path}")
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 70))
            column, page_row = page_offset % args.columns, page_offset // args.columns
            x0, y0 = column * tile_w, page_row * tile_h
            x = x0 + (tile_w - image.width) // 2
            y = y0 + 66 + (tile_h - 70 - image.height) // 2
            sheet.paste(image, (x, y))
            scene = str(row.get("stage220_scene_claim") or "unknown").replace("_metadata_positive", "")
            score = float(row.get("stage220_lowlight_score") or 0.0)
            mean = float(row.get("stage220_pixel_mean_luma") or 0.0)
            draw.text((x0 + 4, y0 + 4), f"#{global_index:03d} {scene} {row.get('color','?')}", fill="black")
            draw.text((x0 + 4, y0 + 22), f"mean={mean:.1f} score={score:.2f} N={row.get('stage220_frame_night_candidate','false')}", fill="black")
            draw.text((x0 + 4, y0 + 40), str(row.get("sha256") or "")[:20], fill="black")
            row_index.append({
                "index": global_index,
                "sheet": sheet_index,
                "sha256": row.get("sha256"),
                "image_path": str(path),
                "color": row.get("color"),
                "scene_claim": row.get("stage220_scene_claim"),
                "mean_luma": mean,
                "lowlight_score": score,
                "frame_lowlight_proxy": row.get("stage220_frame_lowlight_proxy"),
                "frame_night_candidate": row.get("stage220_frame_night_candidate"),
            })
        output = args.output_root / f"stage221-contact-sheet-{sheet_index:02d}.png"
        sheet.save(output, format="PNG", optimize=True)
        sheet_records.append({
            "sheet": sheet_index,
            "first_index": start + 1,
            "last_index": start + len(page_rows),
            "path": str(output),
            "sha256": sha256_file(output),
        })

    report = {
        "schema_version": "stage221-adverse-color-contact-sheets-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_agent_visual_review_material_ready",
        "input_manifest": str(manifest),
        "input_manifest_sha256": actual_sha,
        "rows": len(rows),
        "sheets": sheet_records,
        "row_index": row_index,
        "policy": {
            "train_only": True,
            "validation_pixels_opened": 0,
            "test_accessed": False,
            "frozen_video_used": False,
            "labels_modified": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage221-contact-sheet-index.json"
    atomic_json(report_path, report)
    report_path.with_suffix(".json.sha256").write_text(f"{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "sheet_count": len(sheet_records)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
