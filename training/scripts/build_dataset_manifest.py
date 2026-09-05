#!/usr/bin/env python3
"""Build a VCAS detection manifest from an existing YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["car", "bus", "truck", "motorcycle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--dataset-version", default="dataset-pilot-det-v1")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument(
        "--source-type",
        choices=["public_dataset", "authorized_camera", "test_fixture"],
        required=True,
    )
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-url")
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument(
        "--usage-status",
        choices=["approved_noncommercial", "approved_internal"],
        required=True,
    )
    parser.add_argument("--group-csv", type=Path)
    parser.add_argument(
        "--allow-static-unique-groups",
        action="store_true",
        help="use one synthetic group per static image when no group CSV exists",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def load_group_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relative_path", "camera_id", "video_id", "track_group"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"group CSV must contain columns {sorted(required)}")
    return {
        row["relative_path"].replace("\\", "/"): {
            "camera_id": row["camera_id"],
            "video_id": row["video_id"],
            "track_group": row["track_group"],
        }
        for row in rows
    }


def parse_yolo_label(path: Path) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(parts[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: class id {class_id} is invalid")
        center_x, center_y, width, height = (float(value) for value in parts[1:])
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"{path}:{line_number}: normalized box is invalid")
        detections.append(
            {
                "bbox_xyxy_norm": [x1, y1, x2, y2],
                "vehicle_class": CLASS_NAMES[class_id],
                "occluded": False,
                "truncated": False,
            }
        )
    return detections


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    root = args.dataset_root.resolve()
    group_map = load_group_map(args.group_csv)
    samples: list[dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing images/labels directory for split {split}")
        for image_path in sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            relative_path = image_path.relative_to(root).as_posix()
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing label for {image_path}")
            detections = parse_yolo_label(label_path)
            if not detections:
                continue
            group = group_map.get(relative_path)
            if group is None:
                if not args.allow_static_unique_groups:
                    raise ValueError(
                        f"{relative_path} has no group mapping; provide --group-csv or "
                        "--allow-static-unique-groups for independent public images"
                    )
                unique = image_path.stem
                group = {
                    "camera_id": f"static_camera_{unique}",
                    "video_id": f"static_image_{unique}",
                    "track_group": f"static_image_{unique}",
                }
            with Image.open(image_path) as image:
                width, height = image.size
            samples.append(
                {
                    "sample_id": f"{split}_{image_path.stem}",
                    "relative_path": relative_path,
                    "sha256": sha256_file(image_path),
                    "source_id": args.source_id,
                    "split": split,
                    "task": "detection",
                    "group": group,
                    "image": {"width": width, "height": height},
                    "annotations": {"detections": detections},
                }
            )

    if not samples:
        raise RuntimeError("dataset contains no labeled images")
    manifest = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "labels_version": labels["labels_version"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "detection",
        "license_review_status": "approved",
        "split_policy": {
            "group_keys": ["camera_id", "video_id", "track_group"],
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "seed": args.seed,
        },
        "sources": [
            {
                "source_id": args.source_id,
                "name": args.source_name,
                "source_type": args.source_type,
                "license_id": args.license_id,
                **({"license_url": args.license_url} if args.license_url else {}),
                "authorization_ref": args.authorization_ref,
                "usage_status": args.usage_status,
                "attribution_required": args.source_type == "public_dataset",
                "redistribution_allowed": False,
            }
        ],
        "samples": samples,
    }
    write_json(args.output, manifest)
    print(f"PASS: wrote {len(samples)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
