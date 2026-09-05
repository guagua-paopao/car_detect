#!/usr/bin/env python3
"""Fail-closed InaTRC component gate against Stage167 and its unlabeled pool."""

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


TRUE = {"1", "true", "yes", "y"}
BODY_TYPES = {"truck", "heavy_truck"}
FORBIDDEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "stage148", "stage155")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage167", type=Path, required=True)
    parser.add_argument("--expected-stage167-sha256", required=True)
    parser.add_argument("--existing-unlabeled", type=Path, required=True)
    parser.add_argument("--expected-unlabeled-sha256", required=True)
    parser.add_argument("--stage91-all", type=Path, required=True)
    parser.add_argument("--expected-stage91-all-sha256", required=True)
    parser.add_argument("--stage91-report", type=Path, required=True)
    parser.add_argument("--expected-stage91-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-dhash-distance", type=int, default=4)
    parser.add_argument("--minimum-supervised-rows", type=int, default=5000)
    parser.add_argument("--minimum-unlabeled-rows", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE


def identity(row: dict[str, str]) -> str:
    for key in ("crop_sha256", "sha256", "source_sha256", "source_image_sha256"):
        value = str(row.get(key, "")).strip().lower()
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            return value
    return ""


def manifest_dhash(row: dict[str, str]) -> str:
    for key in ("crop_dhash64", "dhash64"):
        value = str(row.get(key, "")).strip().lower()
        if len(value) == 16 and all(ch in "0123456789abcdef" for ch in value):
            return value
    return ""


def group(row: dict[str, str]) -> str:
    return str(row.get("track_group") or row.get("video_id") or row.get("source_frame_id") or row.get("image_path") or "").strip()


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


class HammingIndex:
    """Exact radius<=4 lookup using five disjoint bit blocks."""

    WIDTHS = (13, 13, 13, 13, 12)

    def __init__(self) -> None:
        self.values: list[int] = []
        self.metadata: list[list[tuple[str, str, str]]] = []
        self.by_value: dict[int, int] = {}
        self.buckets: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in self.WIDTHS]

    @classmethod
    def keys(cls, value: int) -> list[int]:
        keys = []
        shift = 64
        for width in cls.WIDTHS:
            shift -= width
            keys.append((value >> shift) & ((1 << width) - 1))
        return keys

    def add(self, value: int, source: str, split: str, row_group: str) -> None:
        existing = self.by_value.get(value)
        metadata = (source, split, row_group)
        if existing is not None:
            self.metadata[existing].append(metadata)
            return
        index = len(self.values)
        self.by_value[value] = index
        self.values.append(value)
        self.metadata.append([metadata])
        for bucket, key in zip(self.buckets, self.keys(value)):
            bucket[key].append(index)

    def nearest(self, value: int, radius: int) -> tuple[int, list[tuple[str, str, str]]] | None:
        if not 0 <= radius <= 4:
            raise ValueError("five-block index supports radius 0..4")
        candidates: set[int] = set()
        for bucket, key in zip(self.buckets, self.keys(value)):
            candidates.update(bucket.get(key, ()))
        best = None
        for index in candidates:
            distance = (value ^ self.values[index]).bit_count()
            if distance <= radius and (best is None or distance < best[0]):
                best = (distance, self.metadata[index])
        return best


def load_existing(path: Path, source: str, exact: set[str], index: HammingIndex, counters: Counter) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", "train")).strip().lower()
            if split not in {"train", "training", "validation", "val", ""}:
                raise RuntimeError(f"forbidden split in {source}: {split}")
            digest = identity(row)
            if digest:
                exact.add(digest)
            value = manifest_dhash(row)
            if value:
                index.add(int(value, 16), source, split or "train", group(row))
            counters[f"{source}_rows"] += 1
            counters[f"{source}_{split or 'train'}_rows"] += 1


def stage91_row_valid(row: dict[str, str]) -> tuple[bool, str]:
    if str(row.get("split", "")).strip().lower() != "train":
        return False, "nontrain_split"
    if str(row.get("source_dataset", "")).strip() != "InaTRC-v1":
        return False, "source_drift"
    if str(row.get("source_license", "")).strip().upper() != "CC-BY-4.0":
        return False, "license_drift"
    if not truthy(row.get("license_train_eligible")):
        return False, "license_not_train_eligible"
    if str(row.get("color", "")).strip().lower() != "unknown" or truthy(row.get("color_supervised")):
        return False, "fabricated_color"
    supervised = truthy(row.get("body_type_supervised"))
    label = str(row.get("body_type", "")).strip().lower()
    if supervised and label not in BODY_TYPES:
        return False, "invalid_supervised_body"
    if not supervised and label != "unknown":
        return False, "invalid_unlabeled_body"
    return True, ""


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    inputs = (args.stage167, args.existing_unlabeled, args.stage91_all, args.stage91_report)
    if any(marker in str(path).lower() for marker in FORBIDDEN_MARKERS for path in inputs):
        raise RuntimeError("forbidden frozen/test marker in input path")
    hashes = {
        "stage167": require_hash(args.stage167, args.expected_stage167_sha256, "Stage167 manifest"),
        "existing_unlabeled": require_hash(args.existing_unlabeled, args.expected_unlabeled_sha256, "existing unlabeled manifest"),
        "stage91_all": require_hash(args.stage91_all, args.expected_stage91_all_sha256, "Stage91 all manifest"),
        "stage91_report": require_hash(args.stage91_report, args.expected_stage91_report_sha256, "Stage91 report"),
    }
    source_report = json.loads(args.stage91_report.read_text(encoding="utf-8"))
    policy = source_report.get("policy", {})
    if source_report.get("status") != "pass":
        raise RuntimeError("Stage91 source report is not passing")
    if policy.get("source_validation_payloads_read") != 0 or policy.get("source_test_payloads_read") != 0:
        raise RuntimeError("Stage91 publisher holdout payloads were accessed")
    if not policy.get("all_outputs_train_only"):
        raise RuntimeError("Stage91 outputs are not train-only")

    args.output_dir.mkdir(parents=True)
    existing_exact: set[str] = set()
    existing_index = HammingIndex()
    counters = Counter()
    load_existing(args.stage167, "stage167", existing_exact, existing_index, counters)
    load_existing(args.existing_unlabeled, "existing_unlabeled", existing_exact, existing_index, counters)

    accepted_supervised: list[dict[str, str]] = []
    accepted_unlabeled: list[dict[str, str]] = []
    rejected_reasons = Counter()
    accepted_classes = Counter()
    accepted_groups = set()
    exact_seen: set[str] = set()
    stage91_index = HammingIndex()

    with args.stage91_all.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        output_fields = fieldnames + ["stage210_pixel_dhash64", "stage210_role"]
        for row in reader:
            counters["stage91_rows"] += 1
            valid, reason = stage91_row_valid(row)
            if not valid:
                rejected_reasons[reason] += 1
                continue
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.is_file():
                rejected_reasons["missing_image"] += 1
                continue
            if any(marker in str(image_path).lower() for marker in FORBIDDEN_MARKERS):
                raise RuntimeError("forbidden frozen/test marker in Stage91 image path")
            digest = sha256_file(image_path)
            expected = identity(row)
            if not expected or digest != expected:
                rejected_reasons["sha256_mismatch"] += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    value = dhash64(image)
            except Exception:
                rejected_reasons["unreadable_image"] += 1
                continue
            counters["images_opened"] += 1
            if digest in exact_seen:
                rejected_reasons["stage91_exact_duplicate"] += 1
                continue
            exact_seen.add(digest)
            if digest in existing_exact:
                rejected_reasons["already_in_existing_exact"] += 1
                continue
            near = existing_index.nearest(value, args.maximum_dhash_distance)
            if near is not None:
                sources = {source for source, _, _ in near[1]}
                splits = {split for _, split, _ in near[1]}
                if any(split in {"validation", "val"} for split in splits):
                    rejected_reasons["near_existing_validation"] += 1
                elif "existing_unlabeled" in sources:
                    rejected_reasons["near_existing_unlabeled"] += 1
                else:
                    rejected_reasons["near_existing_train"] += 1
                continue
            row_group = group(row)
            internal_near = stage91_index.nearest(value, args.maximum_dhash_distance)
            if internal_near is not None and any(meta_group != row_group for _, _, meta_group in internal_near[1]):
                rejected_reasons["near_stage91_cross_group"] += 1
                continue
            stage91_index.add(value, "stage91", "train", row_group)
            row["stage210_pixel_dhash64"] = f"{value:016x}"
            if truthy(row.get("body_type_supervised")):
                row["stage210_role"] = "supervised_exact_truck_taxonomy"
                accepted_supervised.append(row)
                accepted_classes[str(row.get("body_type")).strip().lower()] += 1
            else:
                row["stage210_role"] = "unlabeled_toll_cctv_feature_consistency"
                accepted_unlabeled.append(row)
            accepted_groups.add(row_group)

    supervised_path = args.output_dir / "attribute_manifest.stage210-inatrc-supervised.csv"
    unlabeled_path = args.output_dir / "attribute_manifest.stage210-inatrc-unlabeled.csv"
    for path, rows in ((supervised_path, accepted_supervised), (unlabeled_path, accepted_unlabeled)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    hard_failures = []
    if rejected_reasons["missing_image"] or rejected_reasons["sha256_mismatch"] or rejected_reasons["unreadable_image"]:
        hard_failures.append("image_integrity_failure")
    if len(accepted_supervised) < args.minimum_supervised_rows:
        hard_failures.append("insufficient_supervised_rows")
    if len(accepted_classes) < 2:
        hard_failures.append("insufficient_supervised_classes")
    if len(accepted_unlabeled) < args.minimum_unlabeled_rows:
        hard_failures.append("insufficient_unlabeled_rows")
    status = "pass_component_ready_for_research_candidate_manifest" if not hard_failures else "fail_closed"

    report = {
        "schema_version": "stage210-inatrc-stage167-component-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputs": {
            "paths": {
                "stage167": str(args.stage167),
                "existing_unlabeled": str(args.existing_unlabeled),
                "stage91_all": str(args.stage91_all),
                "stage91_report": str(args.stage91_report),
            },
            "sha256": hashes,
        },
        "existing": dict(sorted(counters.items())),
        "stage91": {
            "rows": counters["stage91_rows"],
            "images_opened": counters["images_opened"],
            "accepted_supervised_rows": len(accepted_supervised),
            "accepted_unlabeled_rows": len(accepted_unlabeled),
            "accepted_supervised_class_counts": dict(sorted(accepted_classes.items())),
            "accepted_source_groups": len(accepted_groups),
            "rejected_reasons": dict(sorted(rejected_reasons.items())),
        },
        "gate": {
            "minimum_supervised_rows": args.minimum_supervised_rows,
            "minimum_supervised_classes": 2,
            "minimum_unlabeled_rows": args.minimum_unlabeled_rows,
            "maximum_dhash_distance": args.maximum_dhash_distance,
            "hard_failures": hard_failures,
            "training_component_authorized": not hard_failures,
            "authorization_scope": "research-only component merge; not deployment and not final model acceptance",
        },
        "outputs": {
            "supervised_manifest": str(supervised_path),
            "supervised_manifest_sha256": sha256_file(supervised_path),
            "unlabeled_manifest": str(unlabeled_path),
            "unlabeled_manifest_sha256": sha256_file(unlabeled_path),
        },
        "policy": {
            "publisher_train_partition_only": True,
            "publisher_validation_payloads_read": 0,
            "publisher_test_payloads_read": 0,
            "all_stage91_images_reopened_for_integrity": counters["images_opened"] == counters["stage91_rows"],
            "fine_nontruck_labels_fabricated": False,
            "color_labels_fabricated": False,
            "night_truth_claimed": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage210-inatrc-stage167-component-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n"
        f"{sha256_file(supervised_path)}  {supervised_path.name}\n"
        f"{sha256_file(unlabeled_path)}  {unlabeled_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "accepted_supervised_rows": len(accepted_supervised),
        "accepted_unlabeled_rows": len(accepted_unlabeled),
        "class_counts": dict(sorted(accepted_classes.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
