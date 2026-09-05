#!/usr/bin/env python3
"""Build a VCAS attribute Dataset Manifest from an approved annotation CSV."""

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

from src.common import load_json, parse_bool, sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--dataset-version", default="dataset-pilot-attr-v1")
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
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    csv_path = args.attribute_csv.resolve()
    root = csv_path.parent
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("review_status") not in {None, "", "approved"}:
            continue
        image_path = root / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(
            {
                "sample_id": f"attr_{index:06d}_{image_path.stem}",
                "relative_path": row["image_path"].replace("\\", "/"),
                "sha256": sha256_file(image_path),
                "source_id": args.source_id,
                "split": row["split"],
                "task": "attributes",
                "group": {
                    "camera_id": row["camera_id"],
                    "video_id": row["video_id"],
                    "track_group": row["track_group"],
                },
                "image": {"width": width, "height": height},
                "annotations": {
                    "attributes": {
                        "body_type": row["body_type"],
                        "color": row["color"],
                        "crop_quality": row["crop_quality"],
                        "viewpoint": row["viewpoint"],
                        "blur": parse_bool(row["blur"], "blur"),
                        "occluded": parse_bool(row["occluded"], "occluded"),
                        "truncated": parse_bool(row["truncated"], "truncated"),
                        "night": parse_bool(row["night"], "night"),
                    }
                },
            }
        )
    if not samples:
        raise RuntimeError("attribute CSV contains no approved samples")
    manifest = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "labels_version": labels["labels_version"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "attributes",
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
    print(f"PASS: wrote {len(samples)} attribute samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
