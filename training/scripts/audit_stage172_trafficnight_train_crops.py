#!/usr/bin/env python3
"""Independently verify Stage172 TrafficNight crop files, manifest, labels and hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from build_stage70_specialist_manifests import HammingBKTree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = args.manifest.resolve()
    crop_report = args.crop_report.resolve()
    builder = args.builder.resolve()
    images_root = args.images_root.resolve()
    actual = {
        "manifest": sha256_file(manifest),
        "crop_report": sha256_file(crop_report),
        "builder": sha256_file(builder),
    }
    expected = {
        "manifest": args.expected_manifest_sha256.lower(),
        "crop_report": args.expected_crop_report_sha256.lower(),
        "builder": args.expected_builder_sha256.lower(),
    }
    failures: list[str] = []
    if actual != expected:
        failures.append(f"pinned input mismatch: actual={actual} expected={expected}")
    source = json.loads(crop_report.read_text(encoding="utf-8"))
    if source.get("status") != "pass" or source.get("failures"):
        failures.append("source crop report did not pass")
    source_output = source.get("output", {})
    if str(source_output.get("manifest_sha256", "")).lower() != actual["manifest"]:
        failures.append("source report manifest SHA mismatch")
    if Path(str(source_output.get("images", ""))).resolve() != images_root:
        failures.append("source report image-root mismatch")
    if images_root.is_symlink() or not images_root.is_dir():
        failures.append("image root missing or symbolic")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(source_output.get("rows", -1)):
        failures.append(f"row count {len(rows)} != source report {source_output.get('rows')}")

    manifest_paths: set[Path] = set()
    exact_hashes: set[str] = set()
    tree = HammingBKTree()
    counters: Counter[str] = Counter()
    near_duplicate_examples: list[dict[str, object]] = []
    invalid_examples: list[str] = []

    for index, row in enumerate(rows, 2):
        def invalid(reason: str) -> None:
            counters[f"invalid:{reason}"] += 1
            if len(invalid_examples) < 100:
                invalid_examples.append(f"row {index}: {reason}")

        path = Path(row.get("image_path", "")).resolve()
        if not path.is_relative_to(images_root):
            invalid("path outside image root")
            continue
        if path.is_symlink() or not path.is_file():
            invalid("missing or symbolic image")
            continue
        if path in manifest_paths:
            invalid("duplicate image path")
            continue
        manifest_paths.add(path)
        if row.get("split") != "train":
            invalid("non-train split")
        if row.get("color") != "unknown" or row.get("color_supervised") != "false":
            invalid("color truth fabricated")
        if row.get("coarse_body_family", "") not in {"", "car", "truck"}:
            invalid("invalid coarse family")
        fine = row.get("body_type_supervised") == "true"
        if fine and row.get("body_type") not in {"bus", "truck"}:
            invalid("invalid exact body truth")
        if not fine and row.get("body_type") != "unknown":
            invalid("unsupervised row has exact body label")
        if row.get("body_type") == "bus" and row.get("coarse_body_family"):
            invalid("bus incorrectly assigned coarse family")
        if row.get("night") != "true" or row.get("lighting") != "night_source_truth":
            invalid("night source truth missing")
        if row.get("stage172_origin") != "trafficnight_safe_border_quarantine":
            invalid("origin mismatch")

        digest = sha256_file(path)
        if digest != row.get("sha256", "").lower():
            invalid("file SHA256 mismatch")
        if digest in exact_hashes:
            invalid("within-manifest exact duplicate")
        exact_hashes.add(digest)
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                perceptual = dhash64(image)
        except Exception as error:
            invalid(f"image decode failed: {error}")
            continue
        if str(width) != row.get("width") or str(height) != row.get("height"):
            invalid("decoded dimension mismatch")
        if f"{perceptual:016x}" != row.get("dhash64", "").lower():
            invalid("dHash mismatch")
        near = tree.find(perceptual, args.near_duplicate_hamming)
        if near is not None:
            invalid("within-manifest perceptual near duplicate")
            if len(near_duplicate_examples) < 100:
                near_duplicate_examples.append({
                    "left": near.get("image_path"),
                    "right": str(path),
                    "left_label": near.get("source_label"),
                    "right_label": row.get("source_label"),
                })
        tree.add(perceptual, row)
        counters[f"source_label:{row.get('source_label', '')}"] += 1
        if fine:
            counters[f"fine_body:{row.get('body_type', '')}"] += 1
        if row.get("coarse_body_family"):
            counters[f"coarse_family:{row['coarse_body_family']}"] += 1

    filesystem_paths = set(images_root.glob("*.jpg")) if images_root.is_dir() else set()
    unexpected_files = sorted(str(path) for path in filesystem_paths - manifest_paths)
    missing_files = sorted(str(path) for path in manifest_paths - filesystem_paths)
    if unexpected_files:
        failures.append(f"unlisted image files: {len(unexpected_files)}")
    if missing_files:
        failures.append(f"manifest image files missing: {len(missing_files)}")
    if any(key.startswith("invalid:") for key in counters):
        failures.append(f"invalid rows or files: {sum(value for key, value in counters.items() if key.startswith('invalid:'))}")

    return {
        "schema_version": "stage172-trafficnight-train-crops-audit-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": str(manifest), "manifest_sha256": actual["manifest"],
            "crop_report": str(crop_report), "crop_report_sha256": actual["crop_report"],
            "builder": str(builder), "builder_sha256": actual["builder"],
            "images_root": str(images_root),
        },
        "verified": {
            "rows": len(rows),
            "manifest_paths": len(manifest_paths),
            "filesystem_jpegs": len(filesystem_paths),
            "exact_hashes": len(exact_hashes),
            "near_duplicate_hamming": args.near_duplicate_hamming,
            "counters": dict(sorted(counters.items())),
            "unexpected_files": unexpected_files[:100],
            "missing_files": missing_files[:100],
            "near_duplicate_examples": near_duplicate_examples,
            "invalid_examples": invalid_examples,
        },
        "policy": {
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "all_rows_train_only": True,
            "all_colors_unknown": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--crop-report", type=Path, required=True)
    parser.add_argument("--expected-crop-report-sha256", required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["verified"]["rows"], "failures": report["failures"]}, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
