#!/usr/bin/env python3
"""Build a full-scale body repair manifest from training/validation sources only."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "stage148", "stage155")
CLASS_SAMPLE_WEIGHTS = {
    "light_truck": 1.60,
    "van": 1.60,
    "mpv": 1.50,
    "pickup": 1.40,
    "other": 1.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-sha256", required=True)
    parser.add_argument("--ua-report", type=Path, required=True)
    parser.add_argument("--expected-ua-report-sha256", required=True)
    parser.add_argument("--night-manifest", type=Path, required=True)
    parser.add_argument("--expected-night-sha256", required=True)
    parser.add_argument("--night-report", type=Path, required=True)
    parser.add_argument("--expected-night-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-cross-split-dhash", type=int, default=4)
    parser.add_argument("--supplement-night-fraction", type=float, default=0.30)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def identity(row: dict[str, str]) -> str:
    return str(row.get("sha256") or row.get("crop_sha256") or row.get("source_sha256") or "").strip().lower()


def dhash(row: dict[str, str]) -> str:
    return str(row.get("dhash64") or row.get("crop_dhash64") or "").strip().lower()


def group(row: dict[str, str]) -> str:
    return str(row.get("track_group") or row.get("video_id") or row.get("source_frame_id") or row.get("image_path") or "").strip()


def stable_key(namespace: str, row: dict[str, str]) -> str:
    return hashlib.sha256(f"{namespace}|{group(row)}|{identity(row)}|{row.get('image_path', '')}".encode()).hexdigest()


def exact_body(row: dict[str, str]) -> bool:
    return truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown"}


def night(row: dict[str, str]) -> bool:
    lighting = str(row.get("lighting") or "").strip().lower()
    return truthy(row.get("night")) or truthy(row.get("low_light")) or lighting in {
        "night", "low_light", "low-light", "official_night_low_light",
    }


def small(row: dict[str, str]) -> bool:
    return truthy(row.get("small_target")) or str(row.get("vehicle_size") or "").strip().lower() == "small"


def occluded(row: dict[str, str]) -> bool:
    level = str(row.get("occlusion_level") or "").strip().lower()
    return truthy(row.get("occluded")) or truthy(row.get("truncated")) or level not in {"", "0", "none", "clear", "unknown"}


def valid_hex(value: str, width: int) -> bool:
    return len(value) == width and all(ch in "0123456789abcdefABCDEF" for ch in value)


class BKTree:
    """Exact dHash radius index using five disjoint bit blocks.

    Any two 64-bit values within Hamming distance four share at least one of
    five blocks exactly. The buckets provide the candidate superset; the final
    full 64-bit distance check preserves the original BK-tree decision.
    """

    BLOCK_WIDTHS = (13, 13, 13, 13, 12)

    def __init__(self) -> None:
        self.values: list[int] = []
        self.tags: list[set[str]] = []
        self.by_value: dict[int, int] = {}
        self.buckets: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in self.BLOCK_WIDTHS]

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    @classmethod
    def block_keys(cls, value: int) -> list[int]:
        keys: list[int] = []
        shift = 64
        for width in cls.BLOCK_WIDTHS:
            shift -= width
            keys.append((value >> shift) & ((1 << width) - 1))
        return keys

    def add(self, value: int, tag: str) -> None:
        existing = self.by_value.get(value)
        if existing is not None:
            self.tags[existing].add(tag)
            return
        index = len(self.values)
        self.by_value[value] = index
        self.values.append(value)
        self.tags.append({tag})
        for bucket, key in zip(self.buckets, self.block_keys(value)):
            bucket[key].append(index)

    def conflict(self, value: int, radius: int, tag: str) -> bool:
        if not 0 <= radius <= 4:
            raise ValueError("five-block dHash index supports radius 0..4")
        candidates: set[int] = set()
        for bucket, key in zip(self.buckets, self.block_keys(value)):
            candidates.update(bucket.get(key, ()))
        for index in candidates:
            if self.distance(value, self.values[index]) <= radius and any(existing != tag for existing in self.tags[index]):
                return True
        return False


def normalized_image_path(row: dict[str, str], manifest: Path) -> str:
    raw = Path(row.get("image_path", ""))
    resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def ua_admissible(row: dict[str, str]) -> bool:
    if exact_body(row):
        return row.get("body_type") in {"bus", "van"}
    return row.get("body_type") in {"", "unknown"} and row.get("coarse_body_family") == "car"


def lvad_admissible(row: dict[str, str]) -> bool:
    return (
        row.get("split") == "train"
        and not exact_body(row)
        and row.get("body_type") in {"", "unknown"}
        and row.get("coarse_body_family") == "car"
        and row.get("source_label") == "mixed_car_family"
    )


def set_weight(row: dict[str, str], source_multiplier: float) -> None:
    try:
        original = float(row.get("sample_weight") or 1.0)
    except ValueError:
        original = 1.0
    class_multiplier = CLASS_SAMPLE_WEIGHTS.get(row.get("body_type", ""), 1.0)
    row["sample_weight"] = f"{max(0.10, min(10.0, original * source_multiplier * class_multiplier)):.6f}"


def prepare(row: dict[str, str], source_manifest: Path, origin: str, source_multiplier: float) -> dict[str, str]:
    copied = dict(row)
    copied["image_path"] = normalized_image_path(copied, source_manifest)
    copied["stage156_origin"] = origin
    copied["stage156_source_coarse_body_family"] = copied.get("coarse_body_family", "")
    if exact_body(copied):
        copied["coarse_body_family"] = ""
    copied["research_only"] = "true"
    copied["deployment_eligible"] = "false"
    set_weight(copied, source_multiplier)
    return copied


def count_flags(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    counts = {
        "rows": total,
        "night_rows": sum(night(row) for row in rows),
        "small_rows": sum(small(row) for row in rows),
        "occluded_or_truncated_rows": sum(occluded(row) for row in rows),
    }
    for key in ("night", "small", "occluded_or_truncated"):
        counts[f"{key}_fraction"] = counts[f"{key}_rows"] / total if total else 0.0
    return counts


def main() -> int:
    args = parse_args()
    inputs = (args.base_manifest, args.ua_manifest, args.ua_report, args.night_manifest, args.night_report)
    if args.output_root.exists() or Path(str(args.output_root) + ".state.json").exists():
        raise FileExistsError("refusing to overwrite Stage156 evidence")
    if any(marker in str(path).lower() for marker in FROZEN_MARKERS for path in inputs):
        raise RuntimeError("forbidden test/frozen input marker")
    hashes = {
        "base_manifest": require_hash(args.base_manifest, args.expected_base_sha256, "base manifest"),
        "ua_manifest": require_hash(args.ua_manifest, args.expected_ua_sha256, "UA manifest"),
        "ua_report": require_hash(args.ua_report, args.expected_ua_report_sha256, "UA report"),
        "night_manifest": require_hash(args.night_manifest, args.expected_night_sha256, "night manifest"),
        "night_report": require_hash(args.night_report, args.expected_night_report_sha256, "night report"),
    }
    ua_report = json.loads(args.ua_report.read_text(encoding="utf-8"))
    night_report = json.loads(args.night_report.read_text(encoding="utf-8"))
    ua_policy = ua_report.get("policy", {})
    if not (
        ua_report.get("status") == "pass"
        and ua_policy.get("source_partition") == "training_only"
        and ua_policy.get("test_accessed") is False
        and ua_policy.get("frozen_video_used") is False
        and ua_report.get("production_model_modified") is False
    ):
        raise RuntimeError("UA research-only source policy failed")
    night_policy = night_report.get("policy", {})
    if not (
        night_report.get("status") == "pass"
        and night_policy.get("all_outputs_train_only") is True
        and night_policy.get("source_validation_payloads_read") == 0
        and night_policy.get("source_test_payloads_read") == 0
        and night_policy.get("fine_body_labels_fabricated") is False
        and night_policy.get("frozen_video_used") is False
    ):
        raise RuntimeError("night source policy failed")

    base_fields, base_rows = load_csv(args.base_manifest)
    ua_fields, ua_rows = load_csv(args.ua_manifest)
    night_fields, night_rows = load_csv(args.night_manifest)
    rejections: Counter[str] = Counter()
    exact_seen: set[str] = set()
    tree = BKTree()

    validation: list[dict[str, str]] = []
    validation_groups: set[str] = set()
    base_validation = [row for row in base_rows if row.get("split") == "validation"]
    ua_validation = [row for row in ua_rows if row.get("split") == "validation" and exact_body(row) and ua_admissible(row)]
    for source_manifest, origin, rows in (
        (args.base_manifest, "stage83_validation", base_validation),
        (args.ua_manifest, "ua_exact_validation", ua_validation),
    ):
        for source in sorted(rows, key=lambda row: stable_key(f"stage156|{origin}", row)):
            row = prepare(source, source_manifest, origin, 1.0)
            sha, value, tag = identity(row), dhash(row), f"validation:{group(row)}"
            if not valid_hex(sha, 64) or not valid_hex(value, 16):
                rejections[f"{origin}:invalid_hash"] += 1
                continue
            if sha in exact_seen:
                rejections[f"{origin}:exact_duplicate"] += 1
                continue
            number = int(value, 16)
            if tree.conflict(number, args.maximum_cross_split_dhash, tag):
                rejections[f"{origin}:cross_group_dhash_le_{args.maximum_cross_split_dhash}"] += 1
                continue
            exact_seen.add(sha); tree.add(number, tag)
            validation.append(row); validation_groups.add(group(row))

    base_train = [row for row in base_rows if row.get("split") == "train" and exact_body(row)]
    ua_train = [row for row in ua_rows if row.get("split") == "train" and ua_admissible(row)]
    lvad_train = [row for row in night_rows if lvad_admissible(row)]
    ua_night = [row for row in ua_train if night(row)]
    ua_exact_day = [row for row in ua_train if not night(row) and exact_body(row)]
    ua_coarse_day = [row for row in ua_train if not night(row) and not exact_body(row)]
    supplement_night_count = len(ua_night) + len(lvad_train)
    maximum_day = math.floor(supplement_night_count * (1.0 / args.supplement_night_fraction - 1.0))
    remaining_day = max(0, maximum_day - len(ua_exact_day))
    selected_ua = ua_night + ua_exact_day + sorted(
        ua_coarse_day, key=lambda row: stable_key("stage156|ua-coarse-day", row)
    )[:remaining_day]

    train: list[dict[str, str]] = []
    train_sources: Counter[str] = Counter()
    for source_manifest, origin, multiplier, rows in (
        (args.base_manifest, "stage83_full_exact", 1.0, base_train),
        (args.ua_manifest, "ua_cctv_exact_or_coarse", 1.5, selected_ua),
        (args.night_manifest, "lvad_real_night_coarse_car", 1.5, lvad_train),
    ):
        for source in sorted(rows, key=lambda row: stable_key(f"stage156|{origin}", row)):
            row = prepare(source, source_manifest, origin, multiplier)
            sha, value, row_group = identity(row), dhash(row), group(row)
            tag = f"train:{row_group}"
            if row_group in validation_groups:
                rejections[f"{origin}:validation_group_overlap"] += 1
                continue
            if not valid_hex(sha, 64) or not valid_hex(value, 16):
                rejections[f"{origin}:invalid_hash"] += 1
                continue
            if sha in exact_seen:
                rejections[f"{origin}:exact_duplicate_or_validation_overlap"] += 1
                continue
            number = int(value, 16)
            if tree.conflict(number, args.maximum_cross_split_dhash, tag):
                rejections[f"{origin}:cross_group_or_validation_dhash_le_{args.maximum_cross_split_dhash}"] += 1
                continue
            exact_seen.add(sha); tree.add(number, tag)
            train.append(row); train_sources[origin] += 1

    train_groups = {group(row) for row in train}
    validation_sha = {identity(row) for row in validation}
    train_sha = {identity(row) for row in train}
    train_dhash = {dhash(row) for row in train}
    validation_dhash = {dhash(row) for row in validation}
    exact_class_counts = Counter(row["body_type"] for row in train if exact_body(row))
    coarse_counts = Counter(row.get("coarse_body_family", "") for row in train if not exact_body(row))
    supplement = [row for row in train if row.get("stage156_origin") != "stage83_full_exact"]
    full_flags = count_flags(train)
    supplement_flags = count_flags(supplement)
    failures: list[str] = []
    if len(train) < 150000:
        failures.append(f"too few training rows: {len(train)}")
    for label in ("bus", "heavy_truck", "light_truck", "mpv", "other", "pickup", "sedan", "suv", "van"):
        if exact_class_counts[label] < 3000:
            failures.append(f"too few exact {label} rows: {exact_class_counts[label]}")
    if supplement_flags["night_fraction"] < args.supplement_night_fraction:
        failures.append("new repair supplement night fraction below target")
    if supplement_flags["small_fraction"] < 0.20:
        failures.append("new repair supplement small fraction below 0.20")
    if supplement_flags["occluded_or_truncated_fraction"] < 0.15:
        failures.append("new repair supplement occluded/truncated fraction below 0.15")
    if train_groups & validation_groups:
        failures.append("train/validation group overlap")
    if train_sha & validation_sha:
        failures.append("train/validation exact SHA overlap")
    if train_dhash & validation_dhash:
        failures.append("train/validation identical dHash overlap")

    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = args.output_root / "attribute_manifest.stage156-fullscale-body-repair.csv"
    fields = list(base_fields)
    for field in (*ua_fields, *night_fields, "stage156_origin", "stage156_source_coarse_body_family", "research_only", "deployment_eligible", "sample_weight"):
        if field not in fields:
            fields.append(field)
    with manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(train + validation)
    report = {
        "schema_version": "stage156-fullscale-body-repair-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_training_manifest_ready" if not failures else "fail_closed",
        "inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in (
            ("base_manifest", args.base_manifest), ("ua_manifest", args.ua_manifest), ("ua_report", args.ua_report),
            ("night_manifest", args.night_manifest), ("night_report", args.night_report),
        )},
        "output": {
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "train_rows": len(train), "validation_rows": len(validation),
            "exact_train_rows": sum(exact_body(row) for row in train),
            "coarse_train_rows": sum(not exact_body(row) for row in train),
            "train_sources": dict(train_sources), "exact_body_counts": dict(sorted(exact_class_counts.items())),
            "coarse_family_counts": dict(sorted(coarse_counts.items())),
            "full_training_flags": full_flags, "new_repair_supplement_flags": supplement_flags,
        },
        "integrity": {
            "train_validation_group_overlap": len(train_groups & validation_groups),
            "train_validation_exact_sha_overlap": len(train_sha & validation_sha),
            "train_validation_identical_dhash_overlap": len(train_dhash & validation_dhash),
            "maximum_cross_group_dhash": args.maximum_cross_split_dhash,
            "rejections": dict(rejections),
        },
        "policy": {
            "all_inputs_training_or_validation_only": True,
            "research_only": True,
            "deployment_eligible": False,
            "ua_license_isolation_preserved": True,
            "fine_labels_fabricated": False,
            "ambiguous_lvad_truck_bus_excluded": True,
            "test_accessed": False,
            "stage148_test_reused": False,
            "stage155_holdout_reused": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    report_path = args.output_root / "stage156-fullscale-body-repair-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": "stage156-fullscale-body-repair-state-v1",
        "created_at": report["created_at"], "status": report["status"],
        "manifest": str(manifest), "manifest_sha256": report["output"]["manifest_sha256"],
        "report": str(report_path), "report_sha256": sha256_file(report_path),
        **report["policy"],
    }
    state_path = Path(str(args.output_root) + ".state.json")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = args.output_root / "SHA256SUMS"
    sums.write_text(
        f"{sha256_file(manifest)}  {manifest.name}\n{sha256_file(report_path)}  {report_path.name}\n{sha256_file(state_path)}  {state_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **report["output"], "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
