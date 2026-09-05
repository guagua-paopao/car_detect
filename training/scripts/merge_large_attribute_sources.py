#!/usr/bin/env python3
"""Merge trusted attribute sources with cross-source leakage protection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from import_vcor_colors import NearDuplicateIndex, dhash64, sha256_file


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
    "body_type_supervised",
    "color_supervised",
    "annotation_source",
    "source_dataset",
    "source_manifest",
    "source_license",
    "review_method",
    "sha256",
    "dhash64",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="repeatable NAME=/absolute/path/to/attribute_manifest.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-version", default="dataset-large-v1-attributes")
    parser.add_argument("--max-dhash-distance", type=int, default=4)
    return parser.parse_args()


def parse_sources(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --source {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower().replace(" ", "_")
        if not name or name in seen:
            raise ValueError(f"duplicate or empty source name: {name!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        result.append((name, path))
    return result


def supervised(row: dict[str, str], field: str) -> bool:
    value = row.get(f"{field}_supervised", "").strip().lower()
    if not value:
        return row.get("review_status") == "approved"
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid {field}_supervised={value!r}")


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    sources = parse_sources(args.source)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    with args.labels.resolve().open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    body_types = set(labels["body_types"])
    colors = set(labels["colors"])

    candidates: list[tuple[int, int, str, Path, dict[str, str]]] = []
    source_cards: dict[str, dict] = {}
    source_raw_counts: Counter[str] = Counter()
    for source_index, (source_name, manifest) in enumerate(sources):
        card_path = manifest.parent / "dataset_card.json"
        if card_path.is_file():
            with card_path.open("r", encoding="utf-8") as handle:
                source_cards[source_name] = json.load(handle)
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source_raw_counts[source_name] += 1
                if row.get("review_status") != "approved":
                    continue
                if source_name == "human" and row.get("annotation_source") not in {
                    "human",
                    "human_rereview",
                }:
                    continue
                body_supervised = supervised(row, "body_type")
                color_supervised = supervised(row, "color")
                if not body_supervised and not color_supervised:
                    continue
                body_value = row.get("body_type", "unknown")
                color_value = row.get("color", "unknown")
                if body_value not in body_types or color_value not in colors:
                    raise RuntimeError(
                        f"{manifest}: label outside local contract: body={body_value}, color={color_value}"
                    )
                if not body_supervised and body_value != "unknown":
                    raise RuntimeError(f"{manifest}: unsupervised body_type must be unknown")
                if not color_supervised and color_value != "unknown":
                    raise RuntimeError(f"{manifest}: unsupervised color must be unknown")
                image_path = manifest.parent / row["image_path"]
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                split = row["split"]
                split_priority = {"test": 0, "validation": 1, "train": 2}[split]
                human_priority = 0 if source_name == "human" else 1
                candidates.append(
                    (split_priority, human_priority * 1000 + source_index, source_name, manifest, row)
                )
    candidates.sort(key=lambda item: (item[0], item[1], item[4]["image_path"]))

    output_root.mkdir(parents=True)
    exact_hashes: dict[str, tuple[str, str, str]] = {}
    near_index = NearDuplicateIndex(args.max_dhash_distance)
    counters: Counter[str] = Counter()
    rows_out: list[dict[str, str]] = []
    for _, _, source_name, manifest, row in candidates:
        source_path = manifest.parent / row["image_path"]
        digest = sha256_file(source_path)
        if digest in exact_hashes:
            counters["rejected_cross_source_exact_duplicate"] += 1
            previous = exact_hashes[digest]
            if previous[1:] != (row["body_type"], row["color"]):
                counters["rejected_exact_duplicate_label_conflict"] += 1
            continue
        with Image.open(source_path) as source_image:
            source_image.load()
            perceptual = dhash64(source_image.convert("RGB"))
        match = near_index.match(perceptual)
        if match is not None:
            counters["rejected_cross_source_near_duplicate"] += 1
            if match[1] != row["split"]:
                counters["rejected_cross_split_near_duplicate"] += 1
            continue
        exact_hashes[digest] = (source_name, row["body_type"], row["color"])
        near_index.add(perceptual, f"{source_name}:{row['image_path']}", row["split"])
        suffix = source_path.suffix.lower() or ".jpg"
        sample_key = hashlib.sha1(
            f"{source_name}:{row['image_path']}".encode("utf-8")
        ).hexdigest()[:20]
        target_path = output_root / "crops" / row["split"] / f"{source_name}_{sample_key}{suffix}"
        link_or_copy(source_path, target_path)
        body_supervised = supervised(row, "body_type")
        color_supervised = supervised(row, "color")
        source_dataset = row.get("source_dataset") or source_name
        rows_out.append(
            {
                "image_path": target_path.relative_to(output_root).as_posix(),
                "body_type": row["body_type"],
                "color": row["color"],
                "crop_quality": row.get("crop_quality", "usable"),
                "viewpoint": row.get("viewpoint", "unknown"),
                "blur": row.get("blur", "false"),
                "occluded": row.get("occluded", "false"),
                "truncated": row.get("truncated", "false"),
                "night": row.get("night", "false"),
                "camera_id": f"{source_name}:{row['camera_id']}",
                "video_id": f"{source_name}:{row['video_id']}",
                "track_group": f"{source_name}:{row['track_group']}",
                "split": row["split"],
                "source_frame_id": f"{source_name}:{row.get('source_frame_id', row['image_path'])}",
                "review_status": "approved",
                "body_type_supervised": str(body_supervised).lower(),
                "color_supervised": str(color_supervised).lower(),
                "annotation_source": row.get("annotation_source") or "official_dataset",
                "source_dataset": source_dataset,
                "source_manifest": str(manifest),
                "source_license": row.get("source_license", "see source dataset card"),
                "review_method": row.get("review_method") or row.get("annotation_source", "approved"),
                "sha256": digest,
                "dhash64": f"{perceptual:016x}",
            }
        )
        counters[f"accepted_{source_name}"] += 1

    manifest_out = output_root / "attribute_manifest.csv"
    with manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    split_counts = Counter(row["split"] for row in rows_out)
    body_counts = Counter(
        f"{row['split']}:{row['body_type']}"
        for row in rows_out
        if row["body_type_supervised"] == "true"
    )
    color_counts = Counter(
        f"{row['split']}:{row['color']}"
        for row in rows_out
        if row["color_supervised"] == "true"
    )
    card = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_version": labels["labels_version"],
        "rows": len(rows_out),
        "split_counts": dict(split_counts),
        "split_body_counts": dict(body_counts),
        "split_color_counts": dict(color_counts),
        "source_raw_counts": dict(source_raw_counts),
        "source_cards": source_cards,
        "audit_counters": dict(counters),
        "merge_policy": {
            "human_source": "only approved human/human_rereview; calibrated_vlm excluded",
            "evaluation_priority": "test then validation then train; human first within a split",
            "duplicate_policy": f"global SHA256 and dHash distance <= {args.max_dhash_distance}",
        },
    }
    with (output_root / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"PASS: merged {len(rows_out)} trusted attribute samples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
