#!/usr/bin/env python3
"""Build a fail-closed, advertisement-disjoint DVM-CAR color pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SOURCE_TO_COLOR = {
    "Black": "black",
    "White": "white",
    "Silver": "silver_gray",
    "Grey": "silver_gray",
    "Red": "red",
    "Blue": "blue",
    "Navy": "blue",
    "Indigo": "blue",
    "Green": "green",
    "Yellow": "yellow_orange",
    "Orange": "yellow_orange",
    "Gold": "yellow_orange",
    "Brown": "brown_beige",
    "Beige": "brown_beige",
    "Bronze": "brown_beige",
    "Purple": "other",
    "Pink": "other",
    "Magenta": "other",
    "Turquoise": "other",
    "Multicolour": "other",
}
TRAIN_QUOTAS = {
    "black": 21000,
    "white": 21000,
    "silver_gray": 21000,
    "red": 21000,
    "blue": 21000,
    "green": 3500,
    "yellow_orange": 4500,
    "brown_beige": 4000,
    "other": 1338,
}
VALIDATION_QUOTAS = {
    "black": 1000,
    "white": 1000,
    "silver_gray": 1000,
    "red": 1000,
    "blue": 1000,
    "green": 300,
    "yellow_orange": 500,
    "brown_beige": 500,
    "other": 300,
}
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
    "source_dataset",
    "source_color",
    "source_license",
    "review_method",
    "sha256",
    "dhash64",
    "width",
    "height",
    "gray_mean",
    "gray_stddev",
    "edge_variance",
    "source_group",
    "source_group_images",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-members-per-group", type=int, default=4)
    parser.add_argument("--min-group-images", type=int, default=2)
    parser.add_argument("--min-dimension", type=int, default=96)
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def parse_member(name: str) -> tuple[str, str] | None:
    path = PurePosixPath(name)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    parts = name.split("/")
    if len(parts) != 8 or parts[0] != "19586296" or parts[1] != "resized_DVM_clean":
        return None
    source_color = parts[6]
    if source_color not in SOURCE_TO_COLOR:
        return None
    tokens = path.stem.split(chr(36) * 2)
    if len(tokens) < 7:
        return None
    # Color is deliberately excluded so conflicting metadata for one advert is detectable.
    group = "/".join([parts[3], parts[4], parts[5], tokens[4], tokens[5]])
    return group, source_color


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if min(values.shape) > 4:
        values = values[2:-2, 2:-2]
    return float(values.var())


def inspect_image(payload: bytes, min_dimension: int) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(io.BytesIO(payload)) as opened:
        opened.load()
        image = opened.convert("RGB")
    width, height = image.size
    if min(width, height) < min_dimension:
        raise ValueError("small_dimension")
    aspect = width / max(1, height)
    if aspect < 0.45 or aspect > 4.0:
        raise ValueError("extreme_aspect")
    stats = ImageStat.Stat(image.convert("L"))
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    if mean < 10.0 or mean > 247.0:
        raise ValueError("extreme_exposure")
    if stddev < 8.0:
        raise ValueError("low_contrast")
    if edges < 18.0:
        raise ValueError("blurred")
    return image, {
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "dhash": dhash64(image),
        "crop_quality": "good" if stddev >= 18.0 and edges >= 60.0 else "usable",
        "blur": edges < 60.0,
        "night": mean < 55.0,
    }


class NearDuplicateIndex:
    def __init__(self, maximum_distance: int) -> None:
        self.maximum_distance = maximum_distance
        self.bands: dict[tuple[str, int, int], list[tuple[int, str]]] = defaultdict(list)

    def match(self, value: int, label: str) -> str | None:
        candidates: set[tuple[int, str]] = set()
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            candidates.update(self.bands.get((label, band_index, band), []))
        for existing, member in candidates:
            if (existing ^ value).bit_count() <= self.maximum_distance:
                return member
        return None

    def add(self, value: int, label: str, member: str) -> None:
        item = (value, member)
        for band_index in range(4):
            band = (value >> (band_index * 16)) & 0xFFFF
            self.bands[(label, band_index, band)].append(item)


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root} or {staging_root}")
    with args.labels.resolve().open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    missing = sorted(set(SOURCE_TO_COLOR.values()) - set(labels["colors"]))
    if missing:
        raise RuntimeError(f"DVM mapping is outside the local color contract: {missing}")
    if sum(TRAIN_QUOTAS.values()) != 118338:
        raise RuntimeError("train quota must remain exactly 118338")

    counters: Counter[str] = Counter()
    group_state: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            parsed = parse_member(info.filename)
            if parsed is None:
                if PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                    counters["ignored_image_member"] += 1
                continue
            group, source_color = parsed
            state = group_state.setdefault(
                group,
                {"colors": set(), "count": 0, "members": []},
            )
            state["colors"].add(source_color)
            state["count"] += 1
            score = stable_score(args.seed, info.filename)
            state["members"].append((score, info.filename))
            state["members"].sort()
            del state["members"][args.max_members_per_group :]

        groups_by_color: dict[str, list[tuple[str, str, int, list[str]]]] = defaultdict(list)
        for group, state in group_state.items():
            if len(state["colors"]) != 1:
                counters["rejected_group_color_conflict"] += 1
                continue
            if state["count"] < args.min_group_images:
                counters["rejected_single_image_group"] += 1
                continue
            source_color = next(iter(state["colors"]))
            canonical = SOURCE_TO_COLOR[source_color]
            members = [member for _, member in state["members"]]
            groups_by_color[canonical].append((group, source_color, state["count"], members))

        for canonical, groups in groups_by_color.items():
            groups.sort(key=lambda item: stable_score(args.seed + 1, item[0]))
            counters[f"eligible_groups:{canonical}"] = len(groups)

        staging_root.mkdir(parents=True)
        exact_hashes: dict[str, str] = {}
        near_index = NearDuplicateIndex(args.max_dhash_distance)
        rows: list[dict[str, str]] = []

        def consume(split: str, quotas: dict[str, int], offsets: dict[str, int]) -> None:
            for canonical, quota in quotas.items():
                accepted = 0
                groups = groups_by_color[canonical]
                position = offsets.get(canonical, 0)
                while position < len(groups) and accepted < quota:
                    group, source_color, group_count, members = groups[position]
                    position += 1
                    accepted_member = None
                    accepted_payload = None
                    accepted_image = None
                    accepted_metrics = None
                    accepted_digest = None
                    for member in members:
                        try:
                            payload = archive.read(member)
                            digest = hashlib.sha256(payload).hexdigest()
                            if digest in exact_hashes:
                                counters["rejected_exact_duplicate"] += 1
                                continue
                            image, metrics = inspect_image(payload, args.min_dimension)
                            # dHash is grayscale and can collapse genuinely different paint
                            # colors. Compare near duplicates within the canonical color;
                            # exact SHA remains global and catches byte-identical conflicts.
                            if near_index.match(metrics["dhash"], canonical) is not None:
                                counters["rejected_near_duplicate"] += 1
                                continue
                        except Exception as exc:
                            counters[f"rejected_{str(exc)}"] += 1
                            continue
                        accepted_member = member
                        accepted_payload = payload
                        accepted_image = image
                        accepted_metrics = metrics
                        accepted_digest = digest
                        break
                    if accepted_member is None:
                        counters["rejected_group_no_usable_member"] += 1
                        continue
                    assert accepted_payload is not None
                    assert accepted_image is not None
                    assert accepted_metrics is not None
                    assert accepted_digest is not None
                    exact_hashes[accepted_digest] = accepted_member
                    near_index.add(accepted_metrics["dhash"], canonical, accepted_member)
                    group_hash = hashlib.sha256(group.encode("utf-8")).hexdigest()[:20]
                    member_hash = hashlib.sha256(accepted_member.encode("utf-8")).hexdigest()[:20]
                    target_dir = staging_root / "crops" / split / canonical / group_hash[:2]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / f"dvm_{member_hash}.jpg"
                    accepted_image.save(target_path, format="JPEG", quality=95, optimize=True)
                    rows.append(
                        {
                            "image_path": target_path.relative_to(staging_root).as_posix(),
                            "body_type": "unknown",
                            "color": canonical,
                            "crop_quality": accepted_metrics["crop_quality"],
                            "viewpoint": "unknown",
                            "blur": str(accepted_metrics["blur"]).lower(),
                            "occluded": "false",
                            "truncated": "false",
                            "night": str(accepted_metrics["night"]).lower(),
                            "camera_id": "dvm_listing_unknown_camera",
                            "video_id": f"dvm_ad_{group_hash}",
                            "track_group": f"dvm_ad_{group_hash}",
                            "split": split,
                            "source_frame_id": accepted_member,
                            "review_status": "approved",
                            "body_type_supervised": "false",
                            "color_supervised": "true",
                            "source_dataset": "DVM-CAR-2.0",
                            "source_color": source_color,
                            "source_license": "CC-BY-NC-4.0;research-only;non-commercial",
                            "review_method": "official_metadata+same-ad-multiframe-consistency+decode+quality+sha256+dhash3",
                            "sha256": accepted_digest,
                            "dhash64": f"{accepted_metrics['dhash']:016x}",
                            "width": str(accepted_metrics["width"]),
                            "height": str(accepted_metrics["height"]),
                            "gray_mean": str(accepted_metrics["gray_mean"]),
                            "gray_stddev": str(accepted_metrics["gray_stddev"]),
                            "edge_variance": str(accepted_metrics["edge_variance"]),
                            "source_group": group,
                            "source_group_images": str(group_count),
                        }
                    )
                    counters[f"accepted:{split}:{canonical}"] += 1
                    accepted += 1
                offsets[canonical] = position
                if accepted != quota:
                    raise RuntimeError(
                        f"{split}:{canonical} accepted {accepted}, required {quota}; "
                        f"eligible groups {len(groups)}"
                    )

        offsets: dict[str, int] = {}
        consume("validation", VALIDATION_QUOTAS, offsets)
        consume("train", TRAIN_QUOTAS, offsets)

    split_counts = Counter(row["split"] for row in rows)
    split_color_counts = Counter(f"{row['split']}:{row['color']}" for row in rows)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["source_group"]].add(row["split"])
    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked_groups:
        raise RuntimeError(f"source-group leakage detected: {len(leaked_groups)}")
    if split_counts["train"] != 118338:
        raise RuntimeError(f"effective train count is {split_counts['train']}, expected 118338")

    manifest_path = staging_root / "attribute_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "attribute-stage68-dvm-color-pool-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "DVM-CAR 2.0",
            "official_url": "https://deepvisualmarketing.github.io/",
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "license": "CC BY-NC 4.0; research-only; non-commercial",
        },
        "mapping": SOURCE_TO_COLOR,
        "train_quotas": TRAIN_QUOTAS,
        "validation_quotas": VALIDATION_QUOTAS,
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "split_color_counts": dict(split_color_counts),
        "source_groups": len(group_splits),
        "source_group_cross_split_overlap": len(leaked_groups),
        "exact_sha_duplicates_accepted": len(rows) - len(exact_hashes),
        "review_policy": "one accepted image per advertisement; at least two source images with consistent metadata color; decode and quality checks; global exact SHA rejection and within-canonical-color dHash<=3 rejection",
        "domain_policy": "listing/studio color pretraining only; CCTV-domain fine-tuning and independent CCTV validation remain mandatory",
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "training_eligibility": "research-only; non-deployable",
        "audit_counters": dict(counters),
    }
    report_path = staging_root / "stage68-dvm-color-pool-report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    hashes = {
        "attribute_manifest.csv": sha256_file(manifest_path),
        "stage68-dvm-color-pool-report.json": sha256_file(report_path),
    }
    with (staging_root / "SHA256SUMS.json").open("w", encoding="utf-8") as handle:
        json.dump(hashes, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(staging_root, output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
