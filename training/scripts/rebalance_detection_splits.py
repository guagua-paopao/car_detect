#!/usr/bin/env python3
"""Create deterministic multilabel-stratified detector splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, write_json  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", default="vcas-det-release-v1")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def read_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for source_split in SPLITS:
        for image_path in sorted((root / "images" / source_split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image_path.name in seen_names:
                raise ValueError(f"duplicate image filename across splits: {image_path.name}")
            seen_names.add(image_path.name)
            label_path = root / "labels" / source_split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing label: {label_path}")
            class_counts: Counter[int] = Counter()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected five YOLO fields"
                    )
                class_counts[int(parts[0])] += 1
            if not class_counts:
                raise ValueError(f"empty label: {label_path}")
            records.append(
                {
                    "image": image_path,
                    "label": label_path,
                    "name": image_path.name,
                    "class_counts": class_counts,
                }
            )
    return records


def assign_records(
    records: list[dict[str, Any]],
    capacities: dict[str, int],
    seed: str,
) -> dict[str, list[dict[str, Any]]]:
    total_images = len(records)
    if sum(capacities.values()) != total_images:
        raise ValueError(
            f"requested split total {sum(capacities.values())} != {total_images}"
        )
    global_class_counts: Counter[int] = Counter()
    for record in records:
        global_class_counts.update(record["class_counts"])
    desired = {
        split: {
            class_id: count * capacities[split] / total_images
            for class_id, count in global_class_counts.items()
        }
        for split in SPLITS
    }
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    assigned_classes: dict[str, Counter[int]] = {
        split: Counter() for split in SPLITS
    }

    def rarity(record: dict[str, Any]) -> tuple[float, str]:
        score = sum(
            count / global_class_counts[class_id]
            for class_id, count in record["class_counts"].items()
        )
        return (-score, stable_key(seed, record["name"]))

    for record in sorted(records, key=rarity):
        candidates = [
            split for split in SPLITS if len(assigned[split]) < capacities[split]
        ]
        if not candidates:
            raise RuntimeError("no split capacity remains")

        def split_score(split: str) -> tuple[float, float, str]:
            class_deficit = sum(
                max(
                    0.0,
                    desired[split][class_id]
                    - assigned_classes[split][class_id],
                )
                * count
                / max(desired[split][class_id], 1.0)
                for class_id, count in record["class_counts"].items()
            )
            size_deficit = (
                capacities[split] - len(assigned[split])
            ) / capacities[split]
            return (
                class_deficit,
                size_deficit,
                stable_key(seed, f"{record['name']}:{split}"),
            )

        selected = max(candidates, key=split_score)
        assigned[selected].append(record)
        assigned_classes[selected].update(record["class_counts"])
    return assigned


def write_yaml(root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/validation",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    (root / "vehicle_det_v1.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    labels = load_json(args.labels)
    class_names = list(labels["vehicle_classes"])
    expected = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]
    if class_names != expected:
        raise RuntimeError(f"unexpected local class order: {class_names}")
    records = read_records(source_root)
    capacities = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    assigned = assign_records(records, capacities, args.seed)
    report_counts: dict[str, Any] = {}
    for split, split_records in assigned.items():
        class_counts: Counter[int] = Counter()
        for record in split_records:
            image_target = output_root / "images" / split / record["name"]
            label_target = (
                output_root / "labels" / split / f"{Path(record['name']).stem}.txt"
            )
            link_or_copy(record["image"], image_target)
            link_or_copy(record["label"], label_target)
            class_counts.update(record["class_counts"])
        report_counts[split] = {
            "images": len(split_records),
            "class_counts": {
                class_names[class_id]: count
                for class_id, count in sorted(class_counts.items())
            },
        }
    write_yaml(output_root, class_names)
    report = {
        "schema_version": "1.0",
        "labels_version": labels["labels_version"],
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "policy": "deterministic_multilabel_stratification",
        "splits": report_counts,
        "class_names": class_names,
    }
    write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
