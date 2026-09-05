#!/usr/bin/env python3
"""Merge audited Vehicle-Rear color tracks into the Stage102 research lineage.

The consumed Stage148 test and the sealed Stage149 future holdout are not inputs.
All Vehicle-Rear rows are deduplicated against the active DVM/CCTV lineage before
the combined train/validation manifest is emitted.
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
PARTIAL_COLOR_GROUPS = {"brown_beige", "yellow_orange"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def parse_dhash(row: dict[str, str]) -> int | None:
    text = str(row.get("stage150_dhash64") or "").strip().lower()
    if len(text) != 16:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def canonical_dhash64(path: Path) -> int:
    """Hash saved pixels with one direction/resampler across every source."""
    with Image.open(path) as opened:
        image = opened.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def attach_canonical_hashes(rows: list[dict[str, str]], manifest_path: Path, workers: int) -> None:
    resolved: list[Path] = []
    for row in rows:
        path = Path(row.get("image_path", ""))
        if not path.is_absolute():
            path = manifest_path.parent / path
        resolved.append(path.resolve())
    unique_paths = list(dict.fromkeys(resolved))

    def inspect(path: Path) -> tuple[Path, int]:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, canonical_dhash64(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        canonical = dict(executor.map(inspect, unique_paths))
    for row, path in zip(rows, resolved):
        row["stage150_dhash64"] = f"{canonical[path]:016x}"


def block_values(value: int) -> tuple[int, ...]:
    return tuple((value >> start) & ((1 << (end - start)) - 1) for start, end in BLOCKS)


class HammingIndex:
    def __init__(self) -> None:
        self.values: list[int] = []
        self.indexes = [defaultdict(list) for _ in BLOCKS]

    def add(self, value: int) -> None:
        position = len(self.values)
        self.values.append(value)
        for block_index, block in enumerate(block_values(value)):
            self.indexes[block_index][block].append(position)

    def matches(self, value: int, maximum_distance: int = 4) -> list[int]:
        if maximum_distance != 4:
            raise ValueError("five-block index is exact only for distance four")
        candidates: set[int] = set()
        for block_index, block in enumerate(block_values(value)):
            candidates.update(self.indexes[block_index].get(block, ()))
        return sorted(
            position
            for position in candidates
            if (value ^ self.values[position]).bit_count() <= maximum_distance
        )

    def near(self, value: int, maximum_distance: int = 4) -> bool:
        return bool(self.matches(value, maximum_distance))


def weight_for_base(row: dict[str, str]) -> str:
    source = str(row.get("source_dataset", "")).lower()
    prior = float(row.get("sample_weight") or 1.0)
    if "dvm" in source:
        return f"{min(prior, 0.35):.6f}"
    if row.get("split") == "train" and row.get("color", "unknown") != "unknown":
        return f"{max(prior, 2.0):.6f}"
    return f"{prior:.6f}"


def weight_for_vehicle_rear(row: dict[str, str]) -> str:
    prior = float(row.get("sample_weight") or 1.0)
    return f"{min(6.0, prior * 2.0):.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--supplement-manifest", type=Path, required=True)
    parser.add_argument("--supplement-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-supplement-train", type=int, default=3500)
    parser.add_argument("--minimum-total-train", type=int, default=118338)
    parser.add_argument("--minimum-validation", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage150 manifest evidence")
    for path in (args.base_manifest, args.supplement_manifest, args.supplement_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_sha = sha256_file(args.base_manifest)
    if base_sha != args.expected_base_sha256.lower():
        raise RuntimeError("Stage102 base manifest SHA256 mismatch")
    supplement_report = json.loads(args.supplement_report.read_text(encoding="utf-8"))
    if supplement_report.get("status") != "pass_research_only":
        raise RuntimeError("Stage149 supplement audit did not pass")
    if supplement_report.get("policy", {}).get("future_holdout_images_opened") is not False:
        raise RuntimeError("Stage149 future holdout was not sealed")
    if supplement_report.get("policy", {}).get("stage148_test_reused") is not False:
        raise RuntimeError("Stage148 test reuse policy failed")
    if supplement_report.get("output", {}).get("manifest_sha256") != sha256_file(args.supplement_manifest):
        raise RuntimeError("Stage149 supplement manifest SHA256 mismatch")

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
    with args.supplement_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        supplement = list(csv.DictReader(handle))
    for row in base + supplement:
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found")
    if any(row.get("split") not in {"train", "validation"} for row in supplement):
        raise RuntimeError("Stage149 audit manifest contains a forbidden split")
    attach_canonical_hashes(base, args.base_manifest, args.workers)
    attach_canonical_hashes(supplement, args.supplement_manifest, args.workers)

    base_train = [row for row in base if row.get("split") == "train"]
    legacy_base_validation = [row for row in base if row.get("split") == "validation"]
    if len(base_train) < args.minimum_total_train or not legacy_base_validation:
        raise RuntimeError("Stage102 base split contract failed")

    base_exact = {value for row in base_train if (value := parse_hash(row))}
    base_index = HammingIndex()
    for row in base_train:
        value = parse_dhash(row)
        if value is not None:
            base_index.add(value)

    legacy_validation_index = HammingIndex()
    for row in legacy_base_validation:
        if (value := parse_dhash(row)) is not None:
            legacy_validation_index.add(value)
    legacy_base_train_near_validation = sum(
        legacy_validation_index.near(value)
        for row in base_train
        if (value := parse_dhash(row)) is not None
    )

    accepted_supplement: list[dict[str, str]] = []
    rejection = Counter()
    selected_exact: set[str] = set()
    selected_index = HammingIndex()
    for row in sorted(supplement, key=lambda item: (0 if item["split"] == "validation" else 1, item["track_group"], item["source_frame_id"])):
        digest = parse_hash(row)
        perceptual = parse_dhash(row)
        if not digest or perceptual is None:
            rejection["missing_hash"] += 1
            continue
        if digest in base_exact or digest in selected_exact:
            rejection["exact_duplicate"] += 1
            continue
        if base_index.near(perceptual):
            rejection["base_perceptual_near_duplicate"] += 1
            continue
        # Preserve near frames only inside an existing same-vehicle track. A
        # second cross-track near image is excluded by consulting prior rows.
        near_positions = selected_index.matches(perceptual)
        if any(
            accepted_supplement[position]["track_group"] != row["track_group"]
            for position in near_positions
        ):
            rejection["supplement_cross_track_near_duplicate"] += 1
            continue
        copied = dict(row)
        copied["stage149_review_status"] = row.get("review_status", "")
        copied["review_status"] = "approved"
        if copied.get("color") in PARTIAL_COLOR_GROUPS:
            copied["coarse_color_group"] = copied["color"]
            copied["color"] = "unknown"
            copied["color_supervised"] = "false"
        copied["sample_weight"] = weight_for_vehicle_rear(row)
        copied["stage150_origin"] = "vehicle_rear_real_surveillance"
        accepted_supplement.append(copied)
        selected_exact.add(digest)
        selected_index.add(perceptual)

    output: list[dict[str, str]] = []
    for row in base_train:
        copied = dict(row)
        copied["sample_weight"] = weight_for_base(row)
        copied["stage150_origin"] = "stage102_replay"
        output.append(copied)
    output.extend(accepted_supplement)

    train_groups = {row.get("track_group", "") for row in output if row.get("split") == "train" and row.get("track_group")}
    validation_groups = {row.get("track_group", "") for row in output if row.get("split") == "validation" and row.get("track_group")}
    train_exact = {parse_hash(row) for row in output if row.get("split") == "train" and parse_hash(row)}
    validation_exact = {parse_hash(row) for row in output if row.get("split") == "validation" and parse_hash(row)}
    validation_index = HammingIndex()
    for row in output:
        if row.get("split") == "validation" and (value := parse_dhash(row)) is not None:
            validation_index.add(value)
    residual_near = sum(
        validation_index.near(value)
        for row in output
        if row.get("split") == "train" and (value := parse_dhash(row)) is not None
    )
    supplement_train = [row for row in accepted_supplement if row["split"] == "train"]
    failures = []
    if len(supplement_train) < args.minimum_supplement_train:
        failures.append(f"supplement train rows {len(supplement_train)} < {args.minimum_supplement_train}")
    if sum(row.get("split") == "train" for row in output) < args.minimum_total_train:
        failures.append("total train rows below the required color-sample floor")
    if sum(row.get("split") == "validation" for row in output) < args.minimum_validation:
        failures.append("Vehicle-Rear validation rows below the required floor")
    if train_groups & validation_groups:
        failures.append("train/validation group overlap")
    if train_exact & validation_exact:
        failures.append("train/validation exact SHA overlap")
    if residual_near:
        failures.append("train/validation perceptual near overlap")

    fields: list[str] = []
    for row in output:
        for field in row:
            if field not in fields:
                fields.append(field)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    report = {
        "schema_version": "stage150-vehicle-rear-color-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_only" if not failures else "fail_closed",
        "inputs": {
            "base_manifest": str(args.base_manifest.resolve()),
            "base_manifest_sha256": base_sha,
            "base_rows": len(base),
            "base_train_rows_imported": len(base_train),
            "legacy_base_validation_rows_quarantined": len(legacy_base_validation),
            "supplement_manifest": str(args.supplement_manifest.resolve()),
            "supplement_manifest_sha256": sha256_file(args.supplement_manifest),
            "supplement_rows": len(supplement),
            "supplement_report_sha256": sha256_file(args.supplement_report),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256_file(args.output_manifest),
            "rows": len(output),
            "train_rows": sum(row.get("split") == "train" for row in output),
            "validation_rows": sum(row.get("split") == "validation" for row in output),
            "accepted_supplement_train": len(supplement_train),
            "accepted_supplement_validation": sum(row["split"] == "validation" for row in accepted_supplement),
            "train_color_counts": dict(sorted(Counter(row.get("color", "") for row in output if row.get("split") == "train").items())),
            "weighted_source_mass": dict(sorted({
                origin: round(sum(float(row.get("sample_weight") or 1.0) for row in output if row["stage150_origin"] == origin and row.get("split") == "train"), 4)
                for origin in {row["stage150_origin"] for row in output}
            }.items())),
            "rejections": dict(sorted(rejection.items())),
        },
        "integrity": {
            "legacy_base_train_rows_near_legacy_validation": legacy_base_train_near_validation,
            "train_validation_group_overlap": len(train_groups & validation_groups),
            "train_validation_exact_sha_overlap": len(train_exact & validation_exact),
            "train_validation_dhash_distance_le_4_rows": residual_near,
        },
        "policy": {
            "dvm_replay_weight_cap": 0.35,
            "vehicle_rear_weight_multiplier": 2.0,
            "canonical_dhash_recomputed_from_saved_pixels": True,
            "legacy_stage102_validation_imported": False,
            "legacy_stage102_validation_quarantine_reason": "internal DVM train/validation perceptual overlap discovered by Stage150 full audit",
            "future_holdout_imported": False,
            "stage148_test_reused": False,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable",
        },
        "failures": failures,
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(output), "accepted_supplement_train": len(supplement_train), "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
