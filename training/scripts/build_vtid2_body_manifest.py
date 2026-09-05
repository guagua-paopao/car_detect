#!/usr/bin/env python3
"""Audit VTID2 and build a conservative train-only body-type supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageFilter, ImageStat


EXPECTED_ROOT = "Vehicle Type Image Dataset (Version 2) VTID2"
LABEL_MAP = {"Seden": "sedan", "SUV": "suv", "Pickup": "pickup"}
EXCLUDED_FOLDERS = {"Hatchback": "taxonomy_absent", "Other": "ambiguous_mixed_class"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def normalized_sha(image: Image.Image) -> str:
    value = image.convert("RGB").resize((128, 128))
    return hashlib.sha256(value.tobytes()).hexdigest()


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8))
    pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def quality(image: Image.Image) -> tuple[float, float, float]:
    gray = image.convert("L").resize((128, 128))
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    contrast = math.sqrt(float(stats.var[0]))
    sharpness = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0])
    return mean, contrast, sharpness


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--near-hamming", type=int, default=2)
    parser.add_argument("--near-rgb-mad", type=float, default=1.5)
    parser.add_argument("--min-side", type=int, default=24)
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--minimum-per-class", type=int, default=100)
    args = parser.parse_args()
    expected_sha = args.expected_sha256.strip().lower()
    actual_sha = file_sha256(args.archive)
    if actual_sha != expected_sha:
        raise RuntimeError(f"archive SHA256 mismatch: {actual_sha} != {expected_sha}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"output root is not empty; refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    images_root = args.output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    folder_counts = Counter()
    candidates: list[dict] = []
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        members = archive.infolist()
        counters["archive_entries"] = len(members)
        if any(not safe_member(info.filename) for info in members):
            raise RuntimeError("archive contains unsafe member paths")
        for info in members:
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if len(path.parts) != 3 or path.parts[0] != EXPECTED_ROOT or path.suffix.lower() not in {".jpg", ".jpeg"}:
                counters["rejected_unexpected_member_path"] += 1
                continue
            folder = path.parts[1]
            folder_counts[folder] += 1
            if folder in EXCLUDED_FOLDERS:
                counters[f"excluded_{folder.lower()}_{EXCLUDED_FOLDERS[folder]}"] += 1
                continue
            body_type = LABEL_MAP.get(folder)
            if body_type is None:
                counters["rejected_unknown_folder_label"] += 1
                continue
            try:
                raw = archive.read(info)
                with Image.open(io.BytesIO(raw)) as opened:
                    opened.verify()
                with Image.open(io.BytesIO(raw)) as opened:
                    image = opened.convert("RGB")
                width, height = image.size
                if min(width, height) < args.min_side:
                    counters["rejected_too_small"] += 1
                    continue
                mean, contrast, sharpness = quality(image)
                if mean < 8.0 or mean > 250.0 or contrast < 7.0 or sharpness < 3.0:
                    counters["rejected_visibility_quality"] += 1
                    continue
                thumb = image.resize((32, 32)).tobytes()
                candidates.append({
                    "member": info.filename,
                    "filename": path.name,
                    "folder": folder,
                    "body_type": body_type,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "normalized_sha256": normalized_sha(image),
                    "dhash64": dhash64(image),
                    "thumb": thumb,
                    "width": width,
                    "height": height,
                    "mean": mean,
                    "contrast": contrast,
                    "sharpness": sharpness,
                    "quality_score": contrast + math.log1p(sharpness) * 5.0 + math.log1p(width * height),
                })
            except Exception:
                counters["rejected_unreadable_image"] += 1

        uf = UnionFind(len(candidates))
        counters["quality_accepted_before_dedup"] = len(candidates)
        exact_groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(candidates):
            exact_groups[item["normalized_sha256"]].append(index)
        for indexes in exact_groups.values():
            for index in indexes[1:]:
                uf.union(indexes[0], index)
        counters["exact_normalized_duplicate_rows"] = sum(len(indexes) - 1 for indexes in exact_groups.values())

        # A dHash match alone is unsafe for similar vehicles.  Near duplicates
        # require both dHash proximity and very low RGB mean absolute distance.
        near_pairs = 0
        for left in range(len(candidates)):
            left_item = candidates[left]
            for right in range(left + 1, len(candidates)):
                right_item = candidates[right]
                if (left_item["dhash64"] ^ right_item["dhash64"]).bit_count() > args.near_hamming:
                    continue
                mad = sum(abs(a - b) for a, b in zip(left_item["thumb"], right_item["thumb"])) / len(left_item["thumb"])
                if mad <= args.near_rgb_mad:
                    uf.union(left, right)
                    near_pairs += 1
        counters["near_duplicate_pairs"] = near_pairs
        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(candidates)):
            groups[uf.find(index)].append(index)
        selected: list[dict] = []
        conflict_examples = []
        dedup_group_sizes = []
        for indexes in groups.values():
            dedup_group_sizes.append(len(indexes))
            labels = {candidates[index]["body_type"] for index in indexes}
            if len(labels) != 1:
                counters["rejected_conflicting_duplicate_rows"] += len(indexes)
                if len(conflict_examples) < 20:
                    conflict_examples.append([candidates[index]["member"] for index in indexes])
                continue
            best = max(indexes, key=lambda index: candidates[index]["quality_score"])
            selected.append(candidates[best])
            counters["deduplicated_same_label_rows"] += len(indexes) - 1

        class_counts = Counter(item["body_type"] for item in selected)
        class_inverse_sqrt = {label: 1.0 / math.sqrt(count) for label, count in class_counts.items()}
        normalizer = sum(class_inverse_sqrt.values()) / max(1, len(class_inverse_sqrt))
        rows = []
        for index, item in enumerate(sorted(selected, key=lambda value: (value["body_type"], value["member"]))):
            with Image.open(io.BytesIO(archive.read(item["member"]))) as opened:
                image = opened.convert("RGB")
            destination_dir = images_root / item["body_type"]
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"vtid2_{index:05d}.jpg"
            image.save(destination, format="JPEG", quality=95, optimize=True)
            digest = file_sha256(destination)
            low_light = item["mean"] < 70.0
            small = min(item["width"], item["height"]) < 64 or item["width"] * item["height"] < 6400
            soft = item["sharpness"] < 20.0
            weight = min(3.0, max(1.0, class_inverse_sqrt[item["body_type"]] / normalizer))
            rows.append({
                "image_path": f"{args.output_root.name}/images/{item['body_type']}/{destination.name}",
                "body_type": item["body_type"], "color": "unknown",
                "crop_quality": "approved", "viewpoint": "unknown",
                "blur": str(soft).lower(), "occluded": "false", "truncated": "false",
                "night": str(low_light).lower(), "camera_id": "VTID2-Loei-front-gate",
                "video_id": "VTID2-source-sequence-unknown", "track_group": "",
                "split": "train", "source_frame_id": item["member"],
                "review_status": "approved", "body_type_supervised": "true", "color_supervised": "false",
                "annotation_source": "official_dataset_folder", "source_dataset": "VTID2-Mendeley-v3",
                "source_manifest": str(args.archive), "source_license": "CC-BY-4.0",
                "review_method": "official_folder_plus_fail_closed_decode_quality_exact_and_strict_near_dedup",
                "sha256": digest, "dhash64": f"{item['dhash64']:016x}",
                "occlusion_level": "unknown", "vehicle_size": "small" if small else "medium",
                "lighting": "low_light" if low_light else "daylight", "label_confidence": "high",
                "license_train_eligible": "true", "source_group": "VTID2-Loei-front-gate",
                "color_review_status": "not_applicable", "review_score": "1.0", "review_model": "none",
                "formal_train_eligible": "true", "hard_example_priority": "1" if low_light or small or soft else "0",
                "hard_mining_tags": ";".join(tag for tag, enabled in (("low_light", low_light), ("small_target", small), ("soft", soft)) if enabled) or "normal",
                "hard_score": f"{float(low_light) + float(small) + float(soft):.3f}",
                "photometric_mean": f"{item['mean']:.4f}", "photometric_contrast": f"{item['contrast']:.4f}",
                "photometric_sharpness": f"{item['sharpness']:.4f}", "pseudo_label": "false",
                "pseudo_label_confidence": "", "source_image_id": item["member"],
                "small_target": str(int(small)), "low_light": str(int(low_light)), "soft": str(int(soft)),
                "crop_mean_luma": f"{item['mean']:.4f}", "crop_sharpness": f"{item['sharpness']:.4f}",
                "crop_sha256": digest, "sample_weight": f"{weight:.6f}",
                "training_reason": "CC-BY-4.0 real university-gate CCTV crop with exact official body class",
            })

    manifest = args.output_root / "attribute_supplement.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    accepted_counts = Counter(row["body_type"] for row in rows)
    status = "pass" if len(rows) >= args.minimum_rows and all(
        accepted_counts[label] >= args.minimum_per_class for label in LABEL_MAP.values()
    ) else "fail"
    report = {
        "schema_version": "vtid2-body-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": {
            "dataset": "Vehicle Type Image Dataset (Version 2): VTID2",
            "doi": "10.17632/htsngg9tpc.3",
            "url": "https://data.mendeley.com/datasets/htsngg9tpc/3",
            "license": "CC-BY-4.0",
            "archive": str(args.archive.resolve()),
            "archive_bytes": args.archive.stat().st_size,
            "archive_sha256": actual_sha,
            "official_description_rows": 4356,
            "archive_image_rows": sum(folder_counts.values()),
            "description_archive_count_mismatch": sum(folder_counts.values()) != 4356,
        },
        "policy": {
            "accepted_mapping": LABEL_MAP,
            "excluded_folders": EXCLUDED_FOLDERS,
            "all_rows_train_only": bool(rows) and all(row["split"] == "train" for row in rows),
            "color_supervision_disabled": bool(rows) and all(row["color_supervised"] == "false" for row in rows),
            "model_predictions_used": False,
            "frozen_video_used": False,
            "test_split_used": False,
            "deployment_performed": False,
            "near_duplicate_rule": f"dHash <= {args.near_hamming} and 32x32 RGB MAD <= {args.near_rgb_mad}",
            "minimum_effective_rows": args.minimum_rows,
            "minimum_effective_rows_per_class": args.minimum_per_class,
        },
        "folder_counts": dict(sorted(folder_counts.items())),
        "counters": dict(sorted(counters.items())),
        "accepted_rows": len(rows),
        "accepted_body_counts": dict(sorted(accepted_counts.items())),
        "accepted_low_light_rows": sum(row["low_light"] == "1" for row in rows),
        "accepted_small_rows": sum(row["small_target"] == "1" for row in rows),
        "accepted_soft_rows": sum(row["soft"] == "1" for row in rows),
        "dedup_group_size_summary": {
            "groups": len(dedup_group_sizes),
            "singleton_groups": sum(size == 1 for size in dedup_group_sizes),
            "multi_frame_groups": sum(size > 1 for size in dedup_group_sizes),
            "maximum_group_size": max(dedup_group_sizes, default=0),
            "sizes_top20": sorted(dedup_group_sizes, reverse=True)[:20],
        },
        "duplicate_conflict_examples": conflict_examples,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "decision": "eligible for multi-teacher label-consensus review after visual contact-sheet audit; independent validation remains external",
    }
    report_path = args.output_root / "audit-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
