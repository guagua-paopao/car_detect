#!/usr/bin/env python3
"""Materialize high-confidence BMD-45 COCO boxes as extra type-supervised crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


TYPE_MAP = {
    1: "sedan",       # Sedan
    2: "suv",         # SUV
    3: "mpv",         # MUV
    4: "bus",         # Bus
    5: "heavy_truck", # Truck
    8: "light_truck", # LCV
    9: "bus",         # Mini-bus
    10: "bus",        # Tempo-traveller
    12: "van",        # Van
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--min-width", type=int, default=12)
    parser.add_argument("--min-height", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    if not base_rows:
        raise SystemExit("base manifest is empty")
    fields = list(base_rows[0])
    data = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in data["images"]}
    categories = {int(x["id"]): str(x["name"]) for x in data["categories"]}
    crops: list[dict[str, str]] = []
    skipped = Counter()
    type_counts = Counter()
    args.crop_root.mkdir(parents=True, exist_ok=True)
    cached_image_id: int | None = None
    cached_image: Image.Image | None = None
    annotations = data["annotations"][args.shard_index :: args.shard_count]
    for ann in annotations:
        target = TYPE_MAP.get(int(ann.get("category_id", -1)))
        if target is None:
            skipped[f"unmapped:{categories.get(int(ann.get('category_id', -1)), 'unknown')}"] += 1
            continue
        image_meta = images.get(int(ann["image_id"]))
        if not image_meta:
            skipped["missing_image_metadata"] += 1
            continue
        source_path = args.images_root / str(image_meta["file_name"])
        if not source_path.exists():
            skipped["missing_image"] += 1
            continue
        x, y, width, height = (float(v) for v in ann["bbox"])
        if width < args.min_width or height < args.min_height:
            skipped["tiny_box"] += 1
            continue
        image_id = int(ann["image_id"])
        if cached_image_id != image_id:
            if cached_image is not None:
                cached_image.close()
            cached_image = Image.open(source_path).convert("RGB")
            cached_image_id = image_id
        image = cached_image
        assert image is not None
        image_width, image_height = image.size
        left = max(0, math.floor(x - width * args.margin))
        top = max(0, math.floor(y - height * args.margin))
        right = min(image_width, math.ceil(x + width * (1.0 + args.margin)))
        bottom = min(image_height, math.ceil(y + height * (1.0 + args.margin)))
        if right - left < args.min_width or bottom - top < args.min_height:
            skipped["tiny_crop"] += 1
            continue
        crop_name = f"bmdraw_{image_id}_{int(ann['id'])}.jpg"
        rel_dir = "crops/train"
        out_path = args.crop_root / rel_dir / crop_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            image.crop((left, top, right, bottom)).save(
                out_path, format="JPEG", quality=94, optimize=True
            )
        crop_area = (right - left) * (bottom - top)
        raw_row = {key: "" for key in fields}
        raw_row.update({
            "image_path": str(Path("..") / args.crop_root.name / rel_dir / crop_name).replace("\\", "/"),
            "body_type": target,
            "color": "unknown",
            "crop_quality": "raw_bbox_margin08",
            "viewpoint": "unknown",
            "blur": "unknown",
            "occluded": "unknown",
            "truncated": "unknown",
            "night": "unknown",
            "camera_id": "BMD-45",
            "video_id": f"BMD-45:train:{Path(str(image_meta['file_name'])).parent.name}",
            "track_group": f"bmdraw_image_{int(ann['image_id'])}",
            "split": "train",
            "source_frame_id": str(image_meta["file_name"]),
            "review_status": "approved",
            "body_type_supervised": "true",
            "color_supervised": "false",
            "annotation_source": "BMD-45 COCO category mapping",
            "source_dataset": "BMD-45-RAW",
            "source_manifest": str(args.coco),
            "source_license": "CC-BY-4.0",
            "review_method": "deterministic source-category mapping; no color pseudo-label",
            "sha256": "",
            "dhash64": "",
            "occlusion_level": "unknown",
            "vehicle_size": "small" if crop_area < 12288 else "medium" if crop_area < 49152 else "large",
            "lighting": "unknown",
            "label_confidence": "high",
            "license_train_eligible": "true",
            "source_group": "BMD-45-RAW-COCO",
            "color_review_status": "not_supervised",
            "review_score": "1.0",
            "review_model": "source_category_mapping_v23",
            "formal_train_eligible": "true",
            "hard_example_priority": "medium",
            "hard_mining_tags": "raw_detection_crop",
            "hard_score": "1.0",
            "photometric_mean": "",
            "photometric_contrast": "",
            "photometric_sharpness": "",
        })
        crops.append(raw_row)
        type_counts[target] += 1

    if cached_image is not None:
        cached_image.close()

    merged = base_rows + crops
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "schema_version": "bmd-raw-attribute-manifest-v23",
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha(args.base_manifest),
        "coco": str(args.coco),
        "coco_sha256": sha(args.coco),
        "rows_before": len(base_rows),
        "raw_crops_added": len(crops),
        "rows_after": len(merged),
        "raw_type_counts": dict(sorted(type_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "mapping": {categories[k]: v for k, v in TYPE_MAP.items()},
        "margin": args.margin,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "policy": "Only explicit BMD-45 source categories with conservative type mappings receive body supervision; raw crops never receive color pseudo-labels; original validation/test rows are preserved.",
        "frozen_video_used": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
