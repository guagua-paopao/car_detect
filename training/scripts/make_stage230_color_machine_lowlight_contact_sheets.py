#!/usr/bin/env python3
"""Render all effective train-only machine low-light color candidates."""

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

from PIL import Image, ImageDraw, ImageOps


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=184)
    parser.add_argument("--per-sheet", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(f"unsafe manifest: {manifest}")
    actual_sha = sha256(manifest)
    if actual_sha.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_sha}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row for row in all_rows
        if str(row.get("split") or "").strip().lower() == "train"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("stage177_dedup_eligible"))
        and truthy(row.get("stage177_effective_representative"))
        and str(row.get("stage177_scene_origin") or "").startswith("machine_single_frame_")
        and str(row.get("stage177_scene_label") or "").strip().lower() == "low_light"
    ]
    rows.sort(key=lambda row: (
        str(row.get("source_dataset") or ""),
        str(row.get("color") or ""),
        -float(row.get("stage177_lowlight_score") or 0.0),
        str(row.get("stage177_pixel_sha256") or row.get("sha256") or ""),
    ))
    if len(rows) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} rows, got {len(rows)}")

    args.output_root.mkdir(parents=True)
    tile_w, tile_h = 300, 250
    sheet_rows = math.ceil(args.per_sheet / args.columns)
    row_index = []
    sheets = []
    for sheet_number, start in enumerate(range(0, len(rows), args.per_sheet), 1):
        page = rows[start:start + args.per_sheet]
        canvas = Image.new("RGB", (args.columns * tile_w, sheet_rows * tile_h), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(page):
            index = start + offset + 1
            path = Path(str(row.get("image_path") or ""))
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise RuntimeError(f"unsafe or missing train image: {path}")
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 78))
            column, line = offset % args.columns, offset // args.columns
            x0, y0 = column * tile_w, line * tile_h
            canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 74 + (tile_h - 78 - image.height) // 2))
            draw.text((x0 + 4, y0 + 4), f"#{index:03d} {str(row.get('source_dataset') or '?')[:24]}", fill="black")
            draw.text((x0 + 4, y0 + 22), f"proposal lowlight | {row.get('color')} | score={float(row.get('stage177_lowlight_score') or 0):.3f}", fill="black")
            draw.text((x0 + 4, y0 + 40), str(row.get("stage177_scene_origin") or "")[:42], fill="black")
            draw.text((x0 + 4, y0 + 57), str(row.get("stage177_group_key") or "")[:42], fill="black")
            row_index.append({
                "index": index,
                "sheet": sheet_number,
                "source_dataset": row.get("source_dataset"),
                "image_path": str(path),
                "pixel_sha256": row.get("stage177_pixel_sha256") or row.get("sha256"),
                "color": row.get("color"),
                "group_key": row.get("stage177_group_key"),
                "scene_origin": row.get("stage177_scene_origin"),
                "mean_luma": float(row.get("stage177_mean_luma") or 0),
                "lowlight_score": float(row.get("stage177_lowlight_score") or 0),
            })
        output = args.output_root / f"stage230-machine-lowlight-contact-sheet-{sheet_number:02d}.png"
        canvas.save(output, format="PNG", optimize=True)
        sheets.append({"sheet": sheet_number, "path": str(output), "sha256": sha256(output)})

    report = {
        "stage": "stage230-color-machine-lowlight-contact-sheets-r1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_agent_visual_review_material_ready",
        "input_manifest": str(manifest),
        "input_manifest_sha256": actual_sha,
        "selected_rows": len(rows),
        "source_counts": dict(sorted(Counter(row.get("source_dataset") or "unknown" for row in rows).items())),
        "color_counts": dict(sorted(Counter(row.get("color") or "unknown" for row in rows).items())),
        "scene_origin_counts": dict(sorted(Counter(row.get("stage177_scene_origin") or "unknown" for row in rows).items())),
        "sheets": sheets,
        "row_index": row_index,
        "policy": {
            "train_only": True,
            "machine_scene_label_is_proposal_not_truth": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "labels_modified": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage230-machine-lowlight-contact-sheet-index.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    report_path.with_suffix(".json.sha256").write_text(f"{sha256(report_path)}  {report_path.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rows": len(rows),
        "source_counts": report["source_counts"], "sheet_count": len(sheets),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
