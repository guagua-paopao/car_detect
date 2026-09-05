#!/usr/bin/env python3
"""Build a leakage-safe color train/validation split for a fresh lineage.

Stage150 already consumed every DVM training group, so a clean fine-color
validation view cannot be retrofitted to that checkpoint.  This builder
reserves DVM advertisement groups plus unused CC0 contributor groups, merges
the existing Vehicle-Rear and VFG validation views, and removes every exact or
perceptual near duplicate from the future training split.  The output is only
valid for a model initialized outside the Stage80/103/150 color lineage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48")
FINE_COLORS = ("black", "white", "gray", "silver", "red", "blue", "green", "yellow", "brown", "other")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_for(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("source_image_sha256") or "").strip().lower()


def group_for(row: dict[str, str], origin: str) -> str:
    raw = row.get("track_group") or row.get("track_key") or row.get("source_group") or row.get("video_id") or row.get("source_frame_id")
    return f"{origin}:{raw}"


def parse_existing_dhash(row: dict[str, str]) -> int | None:
    text = str(row.get("stage157_dhash64") or row.get("stage150_dhash64") or "").strip().lower()
    if len(text) != 16:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def canonical_dhash64(path: Path) -> int:
    with Image.open(path) as opened:
        gray = opened.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def resolve_path(row: dict[str, str], dataset_root: Path) -> Path:
    path = Path(row.get("image_path", ""))
    return path if path.is_absolute() else dataset_root / path


def attach_hashes(rows: list[dict[str, str]], dataset_root: Path, workers: int) -> None:
    missing = [row for row in rows if parse_existing_dhash(row) is None]
    paths = [resolve_path(row, dataset_root).resolve() for row in missing]
    unique = list(dict.fromkeys(paths))

    def inspect(path: Path) -> tuple[Path, int]:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, canonical_dhash64(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = dict(executor.map(inspect, unique))
    for row, path in zip(missing, paths):
        row["stage157_dhash64"] = f"{values[path]:016x}"
    for row in rows:
        value = parse_existing_dhash(row)
        if value is None:
            raise RuntimeError("missing canonical dHash after attachment")
        row["stage157_dhash64"] = f"{value:016x}"


def block_values(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


class HammingIndex:
    """Exact radius-four index using the five-block pigeonhole property."""

    def __init__(self) -> None:
        self.values: list[int] = []
        self.groups: list[str] = []
        self.indexes = [defaultdict(list) for _ in BLOCKS]

    def add(self, value: int, group: str) -> None:
        position = len(self.values)
        self.values.append(value)
        self.groups.append(group)
        for block_index, block in enumerate(block_values(value)):
            self.indexes[block_index][block].append(position)

    def matches(self, value: int, maximum_distance: int = 4) -> list[int]:
        if not 0 <= maximum_distance <= 4:
            raise ValueError("five-block index supports exact radii zero through four")
        candidates: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(self.indexes[block_index].get(block, ()))
        return sorted(
            position
            for position in candidates
            if (value ^ self.values[position]).bit_count() <= maximum_distance
        )


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_single_color_groups(rows: list[dict[str, str]], origin: str, target: int, seed: str) -> tuple[set[str], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("color_supervised") == "true" and row.get("color") in FINE_COLORS:
            grouped[group_for(row, origin)].append(row)
    by_color: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for group, items in grouped.items():
        colors = {row.get("color") for row in items}
        if len(colors) == 1:
            color = next(iter(colors))
            by_color[color].append((group, len(items)))
    selected: set[str] = set()
    achieved: dict[str, int] = {}
    for color in FINE_COLORS:
        count = 0
        for group, size in sorted(by_color[color], key=lambda item: stable_key(item[0], seed)):
            selected.add(group)
            count += size
            if count >= target:
                break
        achieved[color] = count
    return selected, achieved


def validate_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found")
        if row.get("split") == "test":
            raise RuntimeError("test row found")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage150-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage150-sha256", required=True)
    parser.add_argument("--fine-cc0-manifest", type=Path, required=True)
    parser.add_argument("--fine-cc0-report", type=Path, required=True)
    parser.add_argument("--vfg-validation-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-training-manifest", type=Path, required=True)
    parser.add_argument("--output-validation-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--dvm-target-per-color", type=int, default=250)
    parser.add_argument("--cc0-target-per-color", type=int, default=8)
    parser.add_argument("--minimum-training-rows", type=int, default=118338)
    parser.add_argument("--minimum-validation-per-color", type=int, default=200)
    parser.add_argument("--internal-train-dhash-distance", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", default="stage157-color-split-r1")
    args = parser.parse_args()
    inputs = (args.stage150_manifest, args.fine_cc0_manifest, args.fine_cc0_report, args.vfg_validation_manifest)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_training_manifest, args.output_validation_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    if sha256_file(args.stage150_manifest) != args.expected_stage150_sha256.lower():
        raise RuntimeError("Stage150 manifest SHA256 mismatch")
    fine_report = json.loads(args.fine_cc0_report.read_text(encoding="utf-8"))
    if fine_report.get("status") != "pass":
        raise RuntimeError("fine CC0 recovery did not pass")
    if fine_report.get("output", {}).get("manifest_sha256") != sha256_file(args.fine_cc0_manifest):
        raise RuntimeError("fine CC0 manifest/report SHA256 mismatch")

    def read(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    stage150 = read(args.stage150_manifest)
    cc0 = read(args.fine_cc0_manifest)
    vfg = read(args.vfg_validation_manifest)
    validate_rows(stage150 + cc0 + vfg)
    attach_hashes(stage150 + cc0 + vfg, args.dataset_root, args.workers)

    stage150_train = [dict(row, stage157_origin="stage150_replay") for row in stage150 if row.get("split") == "train"]
    vehicle_rear_validation = [dict(row, stage157_origin="vehicle_rear_validation") for row in stage150 if row.get("split") == "validation"]
    dvm_rows = [row for row in stage150_train if row.get("source_dataset") == "DVM-CAR-2.0"]
    dvm_groups, dvm_achieved = select_single_color_groups(dvm_rows, "dvm", args.dvm_target_per_color, args.seed)
    cc0_groups, cc0_achieved = select_single_color_groups(cc0, "cc0", args.cc0_target_per_color, args.seed)

    validation_candidates: list[dict[str, str]] = []
    for row in dvm_rows:
        if group_for(row, "dvm") in dvm_groups:
            validation_candidates.append(dict(row, stage157_origin="dvm_reserved_validation"))
    validation_candidates.extend(vehicle_rear_validation)
    validation_candidates.extend(dict(row, stage157_origin="vfg_validation") for row in vfg if row.get("split") == "validation")
    validation_candidates.extend(
        dict(row, stage157_origin="cc0_reserved_validation")
        for row in cc0 if group_for(row, "cc0") in cc0_groups
    )

    # Preserve surveillance validation first, then VFG, DVM and auxiliary CC0.
    priority = {"vehicle_rear_validation": 0, "vfg_validation": 1, "dvm_reserved_validation": 2, "cc0_reserved_validation": 3}
    validation_candidates.sort(key=lambda row: (priority[row["stage157_origin"]], group_for(row, row["stage157_origin"]), row.get("image_path", "")))
    validation: list[dict[str, str]] = []
    validation_exact: dict[str, str] = {}
    validation_index = HammingIndex()
    validation_rejections = Counter()
    for row in validation_candidates:
        group = group_for(row, row["stage157_origin"])
        digest = digest_for(row)
        value = parse_existing_dhash(row)
        if not digest or value is None:
            validation_rejections["missing_hash"] += 1
            continue
        if digest in validation_exact and validation_exact[digest] != group:
            validation_rejections["cross_group_exact_duplicate"] += 1
            continue
        matches = validation_index.matches(value)
        if any(validation_index.groups[position] != group for position in matches):
            validation_rejections["cross_group_dhash_le_4"] += 1
            continue
        copied = dict(row)
        copied["split"] = "validation"
        copied["stage157_group"] = group
        validation.append(copied)
        validation_exact[digest] = group
        validation_index.add(value, group)

    training_candidates: list[dict[str, str]] = []
    for row in stage150_train:
        if row.get("source_dataset") == "DVM-CAR-2.0" and group_for(row, "dvm") in dvm_groups:
            continue
        copied = dict(row)
        copied["stage157_group"] = group_for(row, "stage150")
        training_candidates.append(copied)
    for row in cc0:
        if group_for(row, "cc0") in cc0_groups:
            continue
        copied = dict(row, stage157_origin="cc0_fine_train")
        copied["stage157_group"] = group_for(row, "cc0")
        training_candidates.append(copied)

    training: list[dict[str, str]] = []
    training_rejections = Counter()
    training_exact: dict[str, str] = {}
    training_index = HammingIndex()
    for row in training_candidates:
        group = row["stage157_group"]
        digest = digest_for(row)
        value = parse_existing_dhash(row)
        if not digest or value is None:
            training_rejections["missing_hash"] += 1
            continue
        if digest in validation_exact or validation_index.matches(value):
            training_rejections["validation_exact_or_dhash_le_4"] += 1
            continue
        if digest in training_exact and training_exact[digest] != group:
            training_rejections["cross_group_exact_duplicate"] += 1
            continue
        matches = training_index.matches(value, args.internal_train_dhash_distance)
        if any(training_index.groups[position] != group for position in matches):
            training_rejections[f"cross_group_dhash_le_{args.internal_train_dhash_distance}"] += 1
            continue
        copied = dict(row)
        copied["split"] = "train"
        training.append(copied)
        training_exact[digest] = group
        training_index.add(value, group)

    train_groups = {row["stage157_group"] for row in training}
    validation_groups = {row["stage157_group"] for row in validation}
    train_exact_set = {digest_for(row) for row in training}
    validation_exact_set = {digest_for(row) for row in validation}
    residual_near = sum(bool(validation_index.matches(parse_existing_dhash(row))) for row in training)
    failures = []
    if len(training) < args.minimum_training_rows:
        failures.append(f"training rows {len(training)} < {args.minimum_training_rows}")
    if train_groups & validation_groups:
        failures.append("train/validation group overlap")
    if train_exact_set & validation_exact_set:
        failures.append("train/validation exact SHA overlap")
    if residual_near:
        failures.append("train/validation dHash<=4 overlap")
    validation_color_counts = Counter(
        row.get("color", "") for row in validation if row.get("color_supervised") == "true"
    )
    for color in FINE_COLORS:
        if dvm_achieved[color] < args.dvm_target_per_color:
            failures.append(f"DVM reserved {color} rows {dvm_achieved[color]} < {args.dvm_target_per_color}")
        if validation_color_counts[color] < args.minimum_validation_per_color:
            failures.append(
                f"final validation {color} rows {validation_color_counts[color]} < {args.minimum_validation_per_color}"
            )
    args.output_training_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_training_manifest, training)
    write_manifest(args.output_validation_manifest, validation)
    report = {
        "schema_version": "stage157-fresh-color-split-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_only" if not failures else "fail_closed",
        "inputs": {
            "stage150_manifest": str(args.stage150_manifest.resolve()),
            "stage150_manifest_sha256": sha256_file(args.stage150_manifest),
            "fine_cc0_manifest": str(args.fine_cc0_manifest.resolve()),
            "fine_cc0_manifest_sha256": sha256_file(args.fine_cc0_manifest),
            "fine_cc0_report_sha256": sha256_file(args.fine_cc0_report),
            "vfg_validation_manifest": str(args.vfg_validation_manifest.resolve()),
            "vfg_validation_manifest_sha256": sha256_file(args.vfg_validation_manifest),
        },
        "selection": {
            "dvm_target_per_color": args.dvm_target_per_color,
            "dvm_rows_selected_before_dedup": dvm_achieved,
            "cc0_target_per_color": args.cc0_target_per_color,
            "cc0_rows_selected_before_dedup": cc0_achieved,
        },
        "output": {
            "training_manifest": str(args.output_training_manifest.resolve()),
            "training_manifest_sha256": sha256_file(args.output_training_manifest),
            "training_rows": len(training),
            "training_color_counts": dict(sorted(Counter(row.get("color", "") for row in training if row.get("color_supervised") == "true").items())),
            "training_source_counts": dict(sorted(Counter(row.get("stage157_origin", "") for row in training).items())),
            "validation_manifest": str(args.output_validation_manifest.resolve()),
            "validation_manifest_sha256": sha256_file(args.output_validation_manifest),
            "validation_rows": len(validation),
            "validation_color_counts": dict(sorted(validation_color_counts.items())),
            "validation_source_counts": dict(sorted(Counter(row.get("stage157_origin", "") for row in validation).items())),
        },
        "rejections": {
            "validation": dict(sorted(validation_rejections.items())),
            "training": dict(sorted(training_rejections.items())),
        },
        "integrity": {
            "train_validation_group_overlap": len(train_groups & validation_groups),
            "train_validation_exact_sha_overlap": len(train_exact_set & validation_exact_set),
            "train_validation_dhash_distance_le_4_rows": residual_near,
        },
        "policy": {
            "research_only": True,
            "current_stage150_candidate_may_use_validation": False,
            "reason": "reserved DVM validation groups were consumed by the Stage80/103/150 lineage",
            "required_initialization": "fresh ImageNet weights outside Stage80/103/150 color lineage",
            "validation_only_for_future_fresh_lineage": True,
            "vfg_role": "validation_only",
            "vehicle_rear_validation_role": "validation_only",
            "cc0_fine_labels_recovered_by_exact_encoded_crop_sha256": True,
            "internal_training_dedup": f"cross-group exact SHA plus dHash<={args.internal_train_dhash_distance}",
            "train_validation_leakage_dedup": "exact SHA plus dHash<=4",
            "test_rows_used": False,
            "stage148_or_stage155_reused": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "training_rows": len(training), "validation_rows": len(validation), "validation_colors": report["output"]["validation_color_counts"], "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
