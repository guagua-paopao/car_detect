#!/usr/bin/env python3
"""Inventory train-only fine-body supervision not exactly present in Stage167.

This is a metadata-only, fail-closed audit. It never opens image pixels and does
not authorize training. Exact-novel rows still require image integrity, visual
semantics, grouped leakage, and perceptual-near-duplicate audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TARGET_BODY = {
    "sedan",
    "suv",
    "mpv",
    "van",
    "truck",
    "bus",
    "light_truck",
    "heavy_truck",
    "pickup",
    "other",
}
TRUE = {"1", "true", "yes", "y", "approved", "accept", "accepted"}
FALSE = {"0", "false", "no", "n", "rejected", "reject"}
HASH_FIELDS = ("crop_sha256", "sha256", "source_sha256", "source_image_sha256")
PATH_FIELDS = ("image_path", "crop_path")
FORBIDDEN_PATH_TOKENS = (
    "test",
    "holdout",
    "independent",
    "vfg7-eval",
    "hard-v2",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def truthy(value: str | None, default: bool = False) -> bool:
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in TRUE


def row_hash(row: dict[str, str]) -> str:
    for field in HASH_FIELDS:
        value = str(row.get(field, "")).strip().lower()
        if HEX64.fullmatch(value):
            return value
    return ""


def row_path(row: dict[str, str]) -> str:
    for field in PATH_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def body_label(row: dict[str, str]) -> str:
    return str(row.get("body_type", "")).strip().lower()


def train_split(row: dict[str, str]) -> bool:
    value = str(row.get("split", "train")).strip().lower()
    return value in {"", "train", "training"}


def supervised_body(row: dict[str, str]) -> bool:
    if "stage192_body_train_eligible" in row:
        return truthy(row.get("stage192_body_train_eligible"))
    if "body_type_supervised" in row:
        return truthy(row.get("body_type_supervised"))
    return False


def formally_eligible(row: dict[str, str]) -> bool:
    if "stage192_body_train_eligible" in row:
        return truthy(row.get("stage192_body_train_eligible"))
    if "formal_train_eligible" in row:
        return truthy(row.get("formal_train_eligible"))
    if "license_train_eligible" in row:
        return truthy(row.get("license_train_eligible"))
    return False


def license_class(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ").replace("-", " ")
    if any(token in normalized for token in ("by nc", "noncommercial", "non commercial")):
        return "research_noncommercial"
    if any(token in normalized for token in ("cc by 4.0", "cc by 4", "cc0", "apache 2.0", "apache 2")):
        return "permissive_or_attribution"
    return "unresolved"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_stage167_keys(path: Path) -> tuple[set[str], set[str], dict]:
    hashes: set[str] = set()
    paths: set[str] = set()
    summary = {
        "rows": 0,
        "train_rows": 0,
        "body_supervised_train_rows": 0,
        "unique_hashes": 0,
        "unique_paths": 0,
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            summary["rows"] += 1
            if not train_split(row):
                continue
            summary["train_rows"] += 1
            digest = row_hash(row)
            image_path = row_path(row)
            if digest:
                hashes.add(digest)
            if image_path:
                paths.add(image_path)
            if supervised_body(row) and body_label(row) in TARGET_BODY:
                summary["body_supervised_train_rows"] += 1
    summary["unique_hashes"] = len(hashes)
    summary["unique_paths"] = len(paths)
    return hashes, paths, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage167", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    csv.field_size_limit(64 * 1024 * 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_hashes, baseline_paths, baseline_summary = add_stage167_keys(args.stage167)
    discovered = sorted(args.dataset_root.rglob("*.csv")) + list(args.extra)
    manifests: list[Path] = []
    seen_paths: set[str] = set()
    excluded_paths: list[str] = []
    for path in discovered:
        resolved = str(path.resolve())
        lower = resolved.lower()
        if resolved == str(args.stage167.resolve()):
            continue
        if any(token in lower for token in FORBIDDEN_PATH_TOKENS):
            excluded_paths.append(resolved)
            continue
        if resolved in seen_paths or not path.is_file():
            continue
        seen_paths.add(resolved)
        manifests.append(path)

    per_manifest: dict[str, dict] = {}
    exact_novel: dict[str, dict] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    totals = Counter()

    for manifest in manifests:
        stat = Counter()
        class_counts = Counter()
        novel_class_counts = Counter()
        source_counts = Counter()
        try:
            handle = manifest.open("r", encoding="utf-8-sig", newline="")
        except (OSError, UnicodeError) as exc:
            per_manifest[str(manifest)] = {"error": f"open_failed:{type(exc).__name__}"}
            totals["manifest_open_failures"] += 1
            continue
        with handle:
            try:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if "body_type" not in fields:
                    continue
                for row in reader:
                    stat["rows"] += 1
                    totals["rows_scanned"] += 1
                    if not train_split(row):
                        stat["nontrain_rows_skipped"] += 1
                        continue
                    stat["train_rows"] += 1
                    label = body_label(row)
                    if label not in TARGET_BODY:
                        continue
                    if not supervised_body(row):
                        continue
                    stat["body_supervised_train_rows"] += 1
                    if not formally_eligible(row):
                        stat["formal_gate_rejected"] += 1
                        continue
                    license_value = str(row.get("source_license", "")).strip()
                    lic_class = license_class(license_value)
                    stat[f"license_{lic_class}"] += 1
                    if lic_class != "permissive_or_attribution":
                        continue
                    digest = row_hash(row)
                    image_path = row_path(row)
                    if not digest and not image_path:
                        stat["missing_identity"] += 1
                        continue
                    stat["metadata_eligible"] += 1
                    class_counts[label] += 1
                    source = str(row.get("source_dataset", "unknown")).strip() or "unknown"
                    source_counts[source] += 1
                    identity = f"sha256:{digest}" if digest else f"path:{image_path}"
                    conflicts[identity].add(label)
                    if (digest and digest in baseline_hashes) or image_path in baseline_paths:
                        stat["already_in_stage167_exact"] += 1
                        continue
                    stat["exact_novel_rows"] += 1
                    novel_class_counts[label] += 1
                    existing = exact_novel.get(identity)
                    record = {
                        "identity": identity,
                        "crop_sha256": digest,
                        "image_path": image_path,
                        "body_type": label,
                        "source_dataset": source,
                        "source_license": license_value,
                        "source_group": str(row.get("source_group", row.get("track_group", ""))).strip(),
                        "video_id": str(row.get("video_id", "")).strip(),
                        "lighting": str(row.get("lighting", row.get("illumination", ""))).strip(),
                        "night": str(row.get("night", row.get("confirmed_natural_night", ""))).strip(),
                        "origin_manifest": str(manifest),
                        "review_method": str(row.get("review_method", row.get("teacher_audit_policy", ""))).strip(),
                    }
                    if existing is None:
                        exact_novel[identity] = record
                    elif existing["body_type"] != label:
                        existing["label_conflict"] = True
            except (csv.Error, UnicodeError) as exc:
                stat["parse_failure"] += 1
                stat["parse_failure_type"] = type(exc).__name__
                totals["manifest_parse_failures"] += 1

        if stat["body_supervised_train_rows"] or stat["parse_failure"]:
            per_manifest[str(manifest)] = {
                **dict(stat),
                "metadata_eligible_class_counts": dict(sorted(class_counts.items())),
                "exact_novel_class_counts_before_global_dedup": dict(sorted(novel_class_counts.items())),
                "metadata_eligible_source_counts": dict(sorted(source_counts.items())),
                "manifest_sha256": sha256_file(manifest),
            }

    conflict_ids = {identity for identity, labels in conflicts.items() if len(labels) > 1}
    for identity in list(exact_novel):
        if identity in conflict_ids:
            exact_novel.pop(identity)

    records = sorted(
        exact_novel.values(),
        key=lambda r: (r["source_dataset"], r["body_type"], r["identity"]),
    )
    global_classes = Counter(r["body_type"] for r in records)
    global_sources = Counter(r["source_dataset"] for r in records)
    global_lighting = Counter((r["lighting"] or "unknown").lower() for r in records)

    novel_csv = args.output_dir / "stage209-unused-supervised-exact-novel.csv"
    fieldnames = [
        "identity",
        "crop_sha256",
        "image_path",
        "body_type",
        "source_dataset",
        "source_license",
        "source_group",
        "video_id",
        "lighting",
        "night",
        "origin_manifest",
        "review_method",
    ]
    with novel_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    report = {
        "schema_version": "stage209-unused-supervised-inventory-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_metadata_only_fail_closed",
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "stage167_manifest": str(args.stage167),
            "stage167_manifest_sha256": sha256_file(args.stage167),
            "extra_manifests": [str(path) for path in args.extra],
        },
        "stage167": baseline_summary,
        "scan": {
            "manifests_scanned": len(manifests),
            "manifests_excluded_by_test_holdout_path_guard": len(excluded_paths),
            "excluded_paths": excluded_paths,
            **dict(totals),
        },
        "exact_novel": {
            "unique_rows_after_global_exact_dedup_and_label_conflict_rejection": len(records),
            "class_counts": dict(sorted(global_classes.items())),
            "source_counts": dict(sorted(global_sources.items())),
            "lighting_counts": dict(sorted(global_lighting.items())),
            "conflicting_identities_rejected": len(conflict_ids),
            "manifest": str(novel_csv),
        },
        "per_manifest": per_manifest,
        "policy": {
            "metadata_only": True,
            "image_pixels_opened": False,
            "test_holdout_eval_path_guard": True,
            "nontrain_rows_used": False,
            "exact_novel_is_not_training_authorization": True,
            "required_before_training": [
                "image readability and content SHA256 verification",
                "perceptual near-duplicate audit against Stage167 and isolated validation/test",
                "source/video/track grouped leakage audit",
                "fine-body semantic and license evidence audit",
                "class and scene component quota gate",
            ],
            "test_images_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage209-unused-supervised-inventory.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n"
        f"{sha256_file(novel_csv)}  {novel_csv.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "manifests_scanned": len(manifests),
        "rows_scanned": totals["rows_scanned"],
        "exact_novel_rows": len(records),
        "class_counts": dict(sorted(global_classes.items())),
        "source_counts": dict(sorted(global_sources.items())),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
