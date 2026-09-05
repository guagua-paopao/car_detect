#!/usr/bin/env python3
"""Convert BMD-45 COCO labels and compose them with the frozen VCAS dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BMD_TO_VCAS = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Bus": "bus",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
    "Truck": "truck",
    "LCV": "truck",
    "Two-wheeler": "motorcycle",
    "Three-wheeler": "other",
    "Bicycle": "other",
}
VCAS_CLASSES = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmd-root", type=Path, required=True)
    parser.add_argument("--base-detection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-version", default="dataset-domain-bmd45-v1")
    parser.add_argument("--license-review-ref", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def convert_split(
    source_root: Path,
    staging_root: Path,
    source_split: str,
    target_split: str,
    class_to_id: dict[str, int],
) -> Counter[str]:
    annotation_path = source_root / source_split / "_annotations.coco.json"
    payload: dict[str, Any] = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    unknown = sorted(set(categories.values()) - set(BMD_TO_VCAS))
    if unknown:
        raise RuntimeError(f"unmapped BMD-45 categories: {unknown}")

    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["annotations"]:
        annotations[int(row["image_id"])].append(row)

    image_output = staging_root / "images" / target_split
    label_output = staging_root / "labels" / target_split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()

    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        relative = Path(str(image["file_name"]))
        source_image = source_root / source_split / relative
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        target_name = f"bmd45_{source_split.lower()}_{relative.parent.name}_{relative.name}"
        rows: list[str] = []
        for annotation in annotations.get(image_id, []):
            if int(annotation.get("iscrowd", 0)):
                counters["rejected_crowd_boxes"] += 1
                continue
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1 = max(0.0, min(float(width), x))
            y1 = max(0.0, min(float(height), y))
            x2 = max(0.0, min(float(width), x + box_width))
            y2 = max(0.0, min(float(height), y + box_height))
            if x2 <= x1 or y2 <= y1:
                counters["rejected_invalid_boxes"] += 1
                continue
            source_name = categories[int(annotation["category_id"])]
            target_name_class = BMD_TO_VCAS[source_name]
            class_id = class_to_id[target_name_class]
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            rows.append(
                f"{class_id} {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )
            counters[f"box_{target_name_class}"] += 1
            counters[f"source_box_{source_name}"] += 1
        if not rows:
            counters["rejected_empty_images"] += 1
            continue
        target_image = image_output / target_name
        os.link(source_image, target_image)
        (label_output / f"{Path(target_name).stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
        )
        counters["images"] += 1
        counters["boxes"] += len(rows)
    counters["annotation_sha256"] = sha256_file(annotation_path)  # type: ignore[assignment]
    return counters


def main() -> int:
    args = parse_args()
    bmd_root = args.bmd_root.resolve()
    base_root = args.base_detection_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root} or {staging_root}")
    labels = json.loads(args.labels.resolve().read_text(encoding="utf-8"))
    classes = list(labels["vehicle_classes"])
    if classes != VCAS_CLASSES:
        raise RuntimeError(f"unexpected detector contract: {classes}")
    if not args.license_review_ref.strip():
        raise ValueError("license review reference is required")
    for split in ("train", "validation", "test"):
        if not (base_root / "images" / split).is_dir():
            raise FileNotFoundError(base_root / "images" / split)
        if not (base_root / "labels" / split).is_dir():
            raise FileNotFoundError(base_root / "labels" / split)

    class_to_id = {name: index for index, name in enumerate(classes)}
    staging_root.mkdir(parents=True)
    try:
        train_counts = convert_split(
            bmd_root, staging_root, "BMD-45-Train", "bmd_train", class_to_id
        )
        validation_counts = convert_split(
            bmd_root, staging_root, "BMD-45-Val", "bmd_validation", class_to_id
        )
        aggregate_yaml = staging_root / "vehicle_det_v1.yaml"
        write_yaml(
            aggregate_yaml,
            [
                "path: /",
                "train:",
                f"  - {base_root.as_posix()}/images/train",
                f"  - {output_root.as_posix()}/images/bmd_train",
                "val:",
                f"  - {base_root.as_posix()}/images/validation",
                f"  - {output_root.as_posix()}/images/bmd_validation",
                f"test: {base_root.as_posix()}/images/test",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(classes)],
            ],
        )
        write_yaml(
            staging_root / "bmd45_val.yaml",
            [
                f"path: {output_root.as_posix()}",
                "train: images/bmd_train",
                "val: images/bmd_validation",
                "test: images/bmd_validation",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(classes)],
            ],
        )
        report = {
            "schema_version": "1.0",
            "dataset_version": args.dataset_version,
            "labels_version": labels["labels_version"],
            "source": {
                "name": "BMD-45",
                "root": str(bmd_root),
                "license": "CC-BY-4.0",
                "license_review_ref": args.license_review_ref,
            },
            "base_detection_root": str(base_root),
            "output_root": str(output_root),
            "mapping": BMD_TO_VCAS,
            "policy": {
                "official_train_used_for_training_only": True,
                "official_validation_used_for_validation_only": True,
                "base_test_preserved": True,
                "vehicle_class_reserved_for_runtime_fallback": True,
                "images_materialized_as_hardlinks": True,
            },
            "splits": {
                "bmd_train": dict(train_counts),
                "bmd_validation": dict(validation_counts),
            },
        }
        (staging_root / "dataset_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging_root, output_root)
    except Exception:
        raise
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        (output_root / "dataset_report.json").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    print(f"PASS: prepared BMD-45 domain dataset at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
