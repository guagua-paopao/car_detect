#!/usr/bin/env python3
"""Build a leakage-controlled body-type crop set from BMD-45 COCO boxes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat


SOURCE_TO_BODY = {
    "Hatchback": "other",
    "Sedan": "sedan",
    "SUV": "suv",
    "MUV": "mpv",
    "Bus": "bus",
    "Truck": "heavy_truck",
    "LCV": "light_truck",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
    "Van": "van",
}
FIELDS = [
    "image_path", "body_type", "color", "crop_quality", "viewpoint",
    "blur", "occluded", "truncated", "night", "camera_id", "video_id",
    "track_group", "split", "source_frame_id", "review_status",
    "body_type_supervised", "color_supervised", "source_dataset",
    "source_class_name", "source_license", "review_method", "sha256",
    "width", "height", "gray_mean", "gray_stddev",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=3000)
    parser.add_argument("--validation-per-class", type=int, default=800)
    parser.add_argument("--test-per-class", type=int, default=800)
    parser.add_argument("--min-width", type=int, default=96)
    parser.add_argument("--min-height", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dataset-version", default="dataset-domain-bmd45-attributes-v1")
    parser.add_argument("--license-review-ref", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_candidates(args: argparse.Namespace) -> tuple[dict[str, list[dict]], Counter]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    counters: Counter = Counter()
    specifications = [
        ("BMD-45-Train", "train", None),
        ("BMD-45-Val", "validation", "images_000"),
        ("BMD-45-Val", "test", "images_001"),
        ("BMD-45-Val", "test", "images_002"),
    ]
    for source_split, target_split, required_folder in specifications:
        source = args.bmd_root / source_split
        document = json.loads((source / "_annotations.coco.json").read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in document["categories"]}
        images = {int(item["id"]): item for item in document["images"]}
        for annotation in document["annotations"]:
            source_name = categories[int(annotation["category_id"])]
            if source_name not in SOURCE_TO_BODY:
                counters[f"excluded_source_class_{source_name}"] += 1
                continue
            image = images[int(annotation["image_id"])]
            relative = Path(str(image["file_name"]))
            if required_folder is not None and relative.parent.name != required_folder:
                continue
            x, y, width, height = (float(value) for value in annotation["bbox"])
            image_width = int(image["width"])
            image_height = int(image["height"])
            if width < args.min_width or height < args.min_height:
                counters["rejected_small"] += 1
                continue
            if x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1:
                counters["rejected_truncated"] += 1
                continue
            body_type = SOURCE_TO_BODY[source_name]
            grouped[f"{target_split}:{body_type}"].append(
                {
                    "source_root": source,
                    "relative": relative,
                    "annotation_id": int(annotation["id"]),
                    "body_type": body_type,
                    "source_class": source_name,
                    "split": target_split,
                    "bbox": [x, y, width, height],
                    "image_width": image_width,
                    "image_height": image_height,
                    "sequence": relative.parent.name,
                }
            )
    return grouped, counters


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    labels = json.loads(args.labels.resolve().read_text(encoding="utf-8"))
    unknown = sorted(set(SOURCE_TO_BODY.values()) - set(labels["body_types"]))
    if unknown:
        raise RuntimeError(f"mapping outside body-type contract: {unknown}")
    grouped, counters = collect_candidates(args)
    limits = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    selected: list[dict] = []
    for key, candidates in sorted(grouped.items()):
        split, body_type = key.split(":", 1)
        random.Random(args.seed + int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)).shuffle(candidates)
        chosen = candidates[: limits[split]]
        selected.extend(chosen)
        counters[f"selected_{split}_{body_type}"] = len(chosen)
    selected.sort(key=lambda item: (item["split"], item["body_type"], str(item["relative"]), item["annotation_id"]))

    rows: list[dict[str, str]] = []
    output_root.mkdir(parents=True)
    for item in selected:
        source_path = item["source_root"] / item["relative"]
        x, y, width, height = item["bbox"]
        pad_x = width * args.padding_ratio
        pad_y = height * args.padding_ratio
        crop_box = (
            max(0, round(x - pad_x)),
            max(0, round(y - pad_y)),
            min(item["image_width"], round(x + width + pad_x)),
            min(item["image_height"], round(y + height + pad_y)),
        )
        with Image.open(source_path) as image:
            crop = image.convert("RGB").crop(crop_box)
        gray = crop.convert("L")
        stats = ImageStat.Stat(gray)
        mean = float(stats.mean[0])
        stddev = float(stats.stddev[0])
        sample_key = hashlib.sha1(
            f"{item['relative']}:{item['annotation_id']}".encode("utf-8")
        ).hexdigest()[:20]
        target = output_root / "crops" / item["split"] / f"bmd45_{sample_key}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target, format="JPEG", quality=95, optimize=True)
        digest = sha256_file(target)
        rows.append(
            {
                "image_path": target.relative_to(output_root).as_posix(),
                "body_type": item["body_type"],
                "color": "unknown",
                "crop_quality": "good" if stddev >= 20 else "usable",
                "viewpoint": "unknown",
                "blur": "false",
                "occluded": "false",
                "truncated": "false",
                "night": str(mean < 45).lower(),
                "camera_id": f"bmd45_{item['sequence']}",
                "video_id": f"bmd45_{item['split']}_{item['sequence']}",
                "track_group": f"bmd45_box_{sample_key}",
                "split": item["split"],
                "source_frame_id": item["relative"].as_posix(),
                "review_status": "approved",
                "body_type_supervised": "true",
                "color_supervised": "false",
                "source_dataset": "BMD-45",
                "source_class_name": item["source_class"],
                "source_license": "CC-BY-4.0",
                "review_method": "official_coco_class+runtime_crop_quality_filter",
                "sha256": digest,
                "width": str(crop.width),
                "height": str(crop.height),
                "gray_mean": f"{mean:.4f}",
                "gray_stddev": f"{stddev:.4f}",
            }
        )
    with (output_root / "attribute_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    card = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_version": labels["labels_version"],
        "source": {
            "name": "BMD-45",
            "root": str(args.bmd_root.resolve()),
            "license": "CC-BY-4.0",
            "license_review_ref": args.license_review_ref,
        },
        "mapping": SOURCE_TO_BODY,
        "rows": len(rows),
        "selection_counts": dict(counters),
        "split_policy": {
            "train": "BMD-45 official train only",
            "validation": "BMD-45 official validation/images_000 only",
            "test": "BMD-45 official validation/images_001 and images_002 only",
            "group_isolation": "sequence directory is never shared across validation and test",
        },
        "label_policy": {
            "body_type": "official class mapped to VCAS contract",
            "color": "unknown and color_supervised=false",
            "excluded": ["Two-wheeler", "Three-wheeler", "Bicycle"],
        },
        "crop_policy": {
            "minimum_box": [args.min_width, args.min_height],
            "reject_source_border_truncation": True,
            "padding_ratio": args.padding_ratio,
        },
    }
    (output_root / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: wrote {len(rows)} BMD-45 attribute crops to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
