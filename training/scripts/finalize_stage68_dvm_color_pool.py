#!/usr/bin/env python3
"""Add pixel hashes and remove DVM train rows perceptually close to held-out data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


BLOCK_WIDTHS = (13, 13, 13, 13, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        values = np.asarray(gray, dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


def blocks(value: int) -> tuple[int, ...]:
    result, shift = [], 0
    for width in BLOCK_WIDTHS:
        result.append((value >> shift) & ((1 << width) - 1))
        shift += width
    return tuple(result)


class NearIndex:
    def __init__(self) -> None:
        self.values: list[int] = []
        self.indexes = [defaultdict(list) for _ in BLOCK_WIDTHS]

    def add(self, value: int) -> None:
        index = len(self.values)
        self.values.append(value)
        for block_index, block in enumerate(blocks(value)):
            self.indexes[block_index][block].append(index)

    def matches(self, value: int, maximum_distance: int = 4) -> list[int]:
        candidates: set[int] = set()
        for block_index, block in enumerate(blocks(value)):
            candidates.update(self.indexes[block_index].get(block, ()))
        return sorted(index for index in candidates if (value ^ self.values[index]).bit_count() <= maximum_distance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--removed-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-train-rows", type=int, default=118338)
    args = parser.parse_args()
    for path in (args.output_manifest, args.removed_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not fields:
        raise RuntimeError("empty DVM manifest")

    root = args.input_manifest.parent
    for row in rows:
        path = root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        row["source_sha256"] = row.get("sha256", "")
        row["crop_sha256"] = sha256(path)
        row["crop_dhash64"] = f"{dhash64(path):016x}"
        row["stage68_audit_action"] = "retained"

    heldout_index = NearIndex()
    heldout_sha: set[str] = set()
    heldout_rows = 0
    with args.heldout_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in {"validation", "test"}:
                continue
            heldout_rows += 1
            value = (row.get("crop_sha256") or row.get("sha256") or "").strip().lower()
            if value:
                heldout_sha.add(value)
            heldout_index.add(int(row["dhash64"], 16))

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    rejected: list[dict[str, str]] = []
    kept_train: list[dict[str, str]] = []
    deficits: Counter[str] = Counter()
    near_distance_counts: Counter[int] = Counter()
    for row in train:
        value = int(row["crop_dhash64"], 16)
        matches = heldout_index.matches(value)
        exact = row["crop_sha256"] in heldout_sha
        if matches or exact:
            item = dict(row)
            item["stage68_audit_action"] = "removed_external_heldout_similarity"
            item["stage68_heldout_near_pair_count"] = str(len(matches))
            item["stage68_heldout_exact_sha"] = str(exact).lower()
            rejected.append(item)
            deficits[row["color"]] += 1
            for index in matches:
                near_distance_counts[(value ^ heldout_index.values[index]).bit_count()] += 1
        else:
            kept_train.append(row)

    promoted: list[dict[str, str]] = []
    kept_validation: list[dict[str, str]] = []
    for row in validation:
        color = row["color"]
        value = int(row["crop_dhash64"], 16)
        eligible = (
            deficits[color] > 0
            and row["crop_sha256"] not in heldout_sha
            and not heldout_index.matches(value)
        )
        if eligible:
            row["split"] = "train"
            row["image_path"] = row["image_path"].replace("crops/validation/", "crops/validation/", 1)
            row["stage68_audit_action"] = "promoted_validation_group_to_replace_rejected_train_row"
            promoted.append(row)
            deficits[color] -= 1
        else:
            kept_validation.append(row)
    unresolved = {key: value for key, value in deficits.items() if value}
    output = kept_train + promoted + kept_validation

    output_train = [row for row in output if row["split"] == "train"]
    output_validation = [row for row in output if row["split"] == "validation"]
    train_groups = {row["source_group"] for row in output_train}
    validation_groups = {row["source_group"] for row in output_validation}
    output_train_sha = {row["crop_sha256"] for row in output_train}
    residual_exact = output_train_sha & heldout_sha
    residual_near = sum(bool(heldout_index.matches(int(row["crop_dhash64"], 16))) for row in output_train)
    fields = fields + [name for name in ("source_sha256", "crop_sha256", "crop_dhash64", "stage68_audit_action") if name not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    removed_fields = fields + ["stage68_heldout_near_pair_count", "stage68_heldout_exact_sha"]
    with args.removed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=removed_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rejected)

    checks = {
        "minimum_train_rows": len(output_train) >= args.minimum_train_rows,
        "replacement_deficits_resolved": not unresolved,
        "source_group_cross_split_disjoint": not (train_groups & validation_groups),
        "heldout_exact_sha_disjoint": not residual_exact,
        "heldout_dhash_distance_le_4_disjoint": residual_near == 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = {
        "schema_version": "attribute-stage68-dvm-color-finalization-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "heldout_manifest": str(args.heldout_manifest.resolve()),
        "heldout_manifest_sha256": sha256(args.heldout_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "removed_manifest": str(args.removed_manifest.resolve()),
        "removed_manifest_sha256": sha256(args.removed_manifest),
        "input_train_rows": len(train),
        "removed_train_rows": len(rejected),
        "promoted_validation_groups": len(promoted),
        "output_train_rows": len(output_train),
        "output_validation_rows": len(output_validation),
        "heldout_rows_compared": heldout_rows,
        "removed_near_pair_distance_counts": dict(sorted(near_distance_counts.items())),
        "residual_heldout_exact_sha_overlap": len(residual_exact),
        "residual_heldout_near_train_rows": residual_near,
        "source_group_cross_split_overlap": len(train_groups & validation_groups),
        "unresolved_replacement_deficits": unresolved,
        "checks": checks,
        "policy": {
            "all_output_crop_sha256_and_dhash_recomputed_from_saved_pixels": True,
            "heldout_labels_or_predictions_accessed": False,
            "heldout_membership_or_files_modified": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "training_eligibility": "research-only; non-deployable"
        }
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "removed": len(rejected), "promoted": len(promoted), "train": len(output_train), "residual_near": residual_near}))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
