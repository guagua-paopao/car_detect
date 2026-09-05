#!/usr/bin/env python3
"""Crop vehicles from a YOLO dataset and create an attribute annotation template."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = [
    "image_path",
    "body_type",
    "color",
    "crop_quality",
    "viewpoint",
    "blur",
    "occluded",
    "truncated",
    "night",
    "camera_id",
    "video_id",
    "track_group",
    "split",
    "source_frame_id",
    "review_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=1000)
    parser.add_argument("--min-width", type=int, default=96)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    return parser.parse_args()


def parse_label(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("expected 5 YOLO fields")
    return int(parts[0]), *(float(value) for value in parts[1:])


def main() -> int:
    args = parse_args()
    detection_root = args.detection_root.resolve()
    output_root = args.output_root.resolve()
    crops_root = output_root / "crops"
    rows: list[dict[str, str]] = []
    split_quotas = {
        "train": int(round(args.max_crops * 0.70)),
        "validation": int(round(args.max_crops * 0.15)),
    }
    split_quotas["test"] = args.max_crops - sum(split_quotas.values())

    for split in ("train", "validation", "test"):
        split_count = 0
        if split_quotas[split] <= 0:
            continue
        image_dir = detection_root / "images" / split
        label_dir = detection_root / "labels" / split
        split_crop_dir = crops_root / split
        split_crop_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image_width, image_height = image.size
                for box_index, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines()
                ):
                    if not line.strip():
                        continue
                    _, center_x, center_y, width, height = parse_label(line)
                    pixel_width = width * image_width
                    if pixel_width < args.min_width:
                        continue
                    x1 = (center_x - width / 2) * image_width
                    y1 = (center_y - height / 2) * image_height
                    x2 = (center_x + width / 2) * image_width
                    y2 = (center_y + height / 2) * image_height
                    padding_x = (x2 - x1) * args.padding_ratio
                    padding_y = (y2 - y1) * args.padding_ratio
                    crop_box = (
                        max(0, int(x1 - padding_x)),
                        max(0, int(y1 - padding_y)),
                        min(image_width, int(x2 + padding_x)),
                        min(image_height, int(y2 + padding_y)),
                    )
                    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                        continue
                    crop_id = f"{split}_{image_path.stem}_{box_index:03d}"
                    crop_path = split_crop_dir / f"{crop_id}.jpg"
                    image.crop(crop_box).save(crop_path, quality=95)
                    rows.append(
                        {
                            "image_path": crop_path.relative_to(output_root).as_posix(),
                            "body_type": "",
                            "color": "",
                            "crop_quality": "",
                            "viewpoint": "",
                            "blur": "false",
                            "occluded": "false",
                            "truncated": "false",
                            "night": "false",
                            "camera_id": f"static_camera_{image_path.stem}",
                            "video_id": f"static_image_{image_path.stem}",
                            "track_group": f"static_vehicle_{image_path.stem}_{box_index}",
                            "split": split,
                            "source_frame_id": image_path.stem,
                            "review_status": "pending",
                        }
                    )
                    split_count += 1
                    if split_count >= split_quotas[split]:
                        break
            if split_count >= split_quotas[split]:
                break

    if not rows:
        raise RuntimeError("no crops met the requested minimum width")
    csv_path = output_root / "attribute_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS: wrote {len(rows)} crops with target quotas {split_quotas} "
        f"and annotation template to {csv_path}; "
        "fill blank labels and set review_status=approved before training"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
