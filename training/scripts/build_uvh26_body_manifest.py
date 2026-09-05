#!/usr/bin/env python3
"""Build quality-filtered UVH-26 vehicle crops and a train-only supplement.

UVH-26 is used only for body-type supervision.  Color remains explicitly
unknown, and validation/test rows from the existing manifest are copied
unchanged to prevent leakage or test-set tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quality(image: Image.Image) -> tuple[float, float]:
    gray = image.convert("L")
    mean = float(ImageStat.Stat(gray).mean[0])
    # Variance of a high-pass image is a stable, cheap sharpness proxy.
    hp = gray.filter(ImageFilter.FIND_EDGES)
    sharp = float(ImageStat.Stat(hp).var[0])
    return mean, sharp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-plan", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--base-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--padding", type=float, default=0.06)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, str]] = []
    rejected = Counter()
    with args.crop_plan.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        src = args.image_root / Path(row["file_name"]).name
        if not src.is_file():
            rejected["missing_image"] += 1
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                iw, ih = im.size
                x, y = float(row["bbox_x"]), float(row["bbox_y"])
                w, h = float(row["bbox_w"]), float(row["bbox_h"])
                if w <= 1 or h <= 1:
                    rejected["invalid_bbox"] += 1
                    continue
                pad_x, pad_y = w * args.padding, h * args.padding
                left = max(0, math.floor(x - pad_x))
                top = max(0, math.floor(y - pad_y))
                right = min(iw, math.ceil(x + w + pad_x))
                bottom = min(ih, math.ceil(y + h + pad_y))
                crop = im.crop((left, top, right, bottom))
                cw, ch = crop.size
                if min(cw, ch) < args.min_side:
                    rejected["too_small"] += 1
                    continue
                mean, sharp = quality(crop)
                if not (math.isfinite(mean) and math.isfinite(sharp)):
                    rejected["bad_quality"] += 1
                    continue
                # Keep dark/soft samples: they are the hard cases. Only reject
                # unreadable crops, not difficult-but-valid imagery.
                name = f"uvh26_{int(row['source_image_id']):06d}_{i:05d}.jpg"
                out = args.output_root / name
                crop.save(out, "JPEG", quality=95, optimize=True)
                accepted.append({
                    "image_path": str(Path(args.output_root.name) / name),
                    "source_image_id": row["source_image_id"],
                    "source_frame_id": row["file_name"],
                    "track_group": f"UVH26:{row['source_image_id']}",
                    "source_dataset": "UVH-26",
                    "source_license": "CC-BY-4.0",
                    "body_type": row["body_type"],
                    "color": "unknown",
                    "body_type_supervised": "true",
                    "color_supervised": "false",
                    "review_status": "approved",
                    "split": "train",
                    "small_target": row.get("small_target", "0"),
                    "low_light": str(int(mean < 70)),
                    "soft": str(int(sharp < 35)),
                    "hard_score": f"{float(row.get('small_target', 0) or 0) + (1.0 if mean < 70 else 0.0) + (1.0 if sharp < 35 else 0.0):.3f}",
                    "crop_mean_luma": f"{mean:.3f}",
                    "crop_sharpness": f"{sharp:.3f}",
                    "crop_sha256": sha256(out),
                })
        except Exception:
            rejected["unreadable"] += 1

    # Preserve the original manifest exactly in row content and append only
    # approved train supplements. Missing columns are filled with empty values.
    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        base_reader = csv.DictReader(f)
        base_rows = list(base_reader)
        fields = list(base_reader.fieldnames or [])
    for key in accepted[0].keys() if accepted else []:
        if key not in fields:
            fields.append(key)
    # Keep image_path as the canonical path field expected by the dataset.
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in base_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
        for row in accepted:
            writer.writerow({k: row.get(k, "") for k in fields})

    report = {
        "schema_version": "uvh26-body-manifest-v1",
        "source_dataset": "iisc-aim/UVH-26",
        "source_license": "CC-BY-4.0",
        "frozen_video_used": False,
        "crop_plan_rows": len(rows),
        "accepted_crops": len(accepted),
        "rejected": dict(rejected),
        "body_type_counts": dict(Counter(x["body_type"] for x in accepted)),
        "small_target_count": sum(x["small_target"] == "1" for x in accepted),
        "base_rows": len(base_rows),
        "merged_rows": len(base_rows) + len(accepted),
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
        "crop_sha256_sample": [x["crop_sha256"] for x in accepted[:20]],
    }
    report_path = args.output_manifest.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
