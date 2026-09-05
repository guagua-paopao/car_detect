#!/usr/bin/env python3
"""Build a licensed, train-only hard-scene pool for color feature consistency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from build_stage70_specialist_manifests import HammingBKTree


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def resolve_under(root: Path, raw: str) -> Path:
    root = root.resolve()
    value = Path(raw)
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"image escapes safety root: {resolved}") from exc
    return resolved


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def build(args: argparse.Namespace) -> dict[str, object]:
    actual = {
        "stage78": sha256(args.stage78_manifest),
        "stage61": sha256(args.stage61_manifest),
    }
    if actual["stage78"] != args.expected_stage78_sha256.lower():
        raise RuntimeError("Stage78 immutable SHA256 mismatch")
    if actual["stage61"] != args.expected_stage61_sha256.lower():
        raise RuntimeError("Stage61 immutable SHA256 mismatch")
    base_fields, base_rows = read_manifest(args.stage78_manifest)
    source_fields, source_rows = read_manifest(args.stage61_manifest)
    if any(row.get("split", "").strip().lower() == "test" for row in source_rows):
        raise RuntimeError("Stage61 source contains a test row")

    base_exact = {content_hash(row) for row in base_rows if content_hash(row)}
    base_tree = HammingBKTree()
    for row in base_rows:
        value = perceptual_hash(row)
        if len(value) == 16:
            base_tree.add(int(value, 16), row)

    candidates: list[dict[str, str]] = []
    counters = Counter()
    for row in source_rows:
        if row.get("split", "").strip().lower() != "train":
            counters["nontrain_excluded"] += 1
            continue
        if not truthy(row.get("license_train_eligible")):
            counters["license_ineligible_excluded"] += 1
            continue
        searchable = " ".join(row.values()).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen marker found in Stage61 source")
        path = resolve_under(args.datasets_root, row.get("image_path", ""))
        if not readable(path):
            counters["unreadable_excluded"] += 1
            continue
        digest = content_hash(row)
        value = perceptual_hash(row)
        if digest and digest in base_exact:
            counters["exact_overlap_stage78"] += 1
            continue
        if len(value) == 16 and base_tree.find(int(value, 16), args.near_duplicate_hamming) is not None:
            counters["near_overlap_stage78"] += 1
            continue
        output = dict(row)
        output["image_path"] = str(path)
        output["split"] = "train"
        output["body_type"] = "unknown"
        output["color"] = "unknown"
        output["body_type_supervised"] = "false"
        output["color_supervised"] = "false"
        output["review_status"] = "approved"
        output["pseudo_label"] = "false"
        output["pseudo_label_confidence"] = ""
        output["stage95_policy"] = "unknown_safe_color_feature_consistency_only"
        output["stage95_source_review_status"] = row.get("review_status", "")
        candidates.append(output)

    # Deterministic within-pool exact and perceptual dedup. Conflicting labels
    # cannot occur because both heads were reset to unknown before this step.
    tree = HammingBKTree()
    exact: set[str] = set()
    deduplicated: list[dict[str, str]] = []
    for row in sorted(candidates, key=lambda item: item["image_path"]):
        digest = content_hash(row)
        value = perceptual_hash(row)
        if digest and digest in exact:
            counters["candidate_exact_duplicate"] += 1
            continue
        if len(value) == 16 and tree.find(int(value, 16), args.near_duplicate_hamming) is not None:
            counters["candidate_near_duplicate"] += 1
            continue
        deduplicated.append(row)
        if digest:
            exact.add(digest)
        if len(value) == 16:
            tree.add(int(value, 16), row)

    night_rows = sum(
        truthy(row.get("night"))
        and row.get("lighting", "").strip().lower()
        == "night_machine_photometric_consensus"
        for row in deduplicated
    )
    small_rows = sum(row.get("vehicle_size", "").strip().lower() == "small" for row in deduplicated)
    hard_rows = sum(
        row.get("vehicle_size", "").strip().lower() == "small"
        or truthy(row.get("occluded"))
        or truthy(row.get("truncated"))
        or truthy(row.get("blur"))
        for row in deduplicated
    )
    failures: list[str] = []
    if len(deduplicated) < args.minimum_rows:
        failures.append(f"rows {len(deduplicated)} < {args.minimum_rows}")
    if night_rows < args.minimum_night_rows:
        failures.append(f"verified night rows {night_rows} < {args.minimum_night_rows}")
    if any(
        row.get("split") != "train"
        or truthy(row.get("body_type_supervised"))
        or truthy(row.get("color_supervised"))
        or row.get("body_type") != "unknown"
        or row.get("color") != "unknown"
        for row in deduplicated
    ):
        failures.append("fail-closed unlabeled contract violated")
    report: dict[str, object] = {
        "schema_version": "stage95-color-consistency-manifest-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stage78_manifest": str(args.stage78_manifest.resolve()),
            "stage78_sha256": actual["stage78"],
            "stage61_manifest": str(args.stage61_manifest.resolve()),
            "stage61_sha256": actual["stage61"],
        },
        "output": {
            "rows": len(deduplicated),
            "verified_night_rows": night_rows,
            "small_rows": small_rows,
            "hard_rows": hard_rows,
            "counters": dict(sorted(counters.items())),
        },
        "policy": {
            "all_rows_train_only": True,
            "all_attributes_unknown_unsupervised": True,
            "verified_night_requires_machine_label_and_scene_crop_photometry": True,
            "stage78_train_and_validation_dedup_reference": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
        },
        "failures": failures,
    }
    fields = list(dict.fromkeys([
        *source_fields, *base_fields, "stage95_policy", "stage95_source_review_status",
    ]))
    return {"report": report, "rows": deduplicated, "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage78-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage78-sha256", required=True)
    parser.add_argument("--stage61-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage61-sha256", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-rows", type=int, default=3000)
    parser.add_argument("--minimum-night-rows", type=int, default=1500)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage95 evidence")
    result = build(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["report"]["status"] != "pass":
        return 1
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(result["rows"])
    result["report"]["output"]["manifest"] = str(args.output_manifest.resolve())
    result["report"]["output"]["manifest_sha256"] = sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", **result["report"]["output"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
