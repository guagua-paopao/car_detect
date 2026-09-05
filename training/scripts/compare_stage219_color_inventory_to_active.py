#!/usr/bin/env python3
"""Compare Stage218 authoritative colors with the active Stage157 train pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
FROZEN_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}
ADVERSE_SCENES = {"night_metadata_positive", "low_light_metadata_positive", "low_luminance_proxy"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row: dict[str, str], manifest: Path) -> tuple[str, str]:
    sha = str(row.get("sha256") or row.get("crop_sha256") or row.get("source_sha256") or "").strip().lower()
    if HEX64.fullmatch(sha):
        return "sha256", sha
    raw = str(row.get("image_path") or "").strip()
    image = Path(raw)
    if not image.is_absolute():
        image = manifest.parent / image
    return "path", os.path.normcase(os.path.normpath(str(image)))


def has_frozen_marker(row: dict[str, str]) -> bool:
    text = " ".join(str(row.get(field) or "") for field in (
        "image_path", "source_manifest", "source_frame_id", "video_id",
    )).lower()
    return any(marker in text for marker in FROZEN_MARKERS)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--expected-active-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    inventory = args.inventory.resolve()
    active = args.active_manifest.resolve()
    for path, expected in (
        (inventory, args.expected_inventory_sha256),
        (active, args.expected_active_manifest_sha256),
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"input must be a regular non-symlink file: {path}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    active_keys: set[tuple[str, str]] = set()
    active_train_rows = 0
    with active.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split"}
        if missing := required - set(reader.fieldnames or []):
            raise RuntimeError(f"active manifest missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            if str(row.get("split") or "").strip().lower() != "train":
                continue
            if has_frozen_marker(row):
                raise RuntimeError(f"frozen marker in active train row {row_number}")
            active_train_rows += 1
            active_keys.add(row_key(row, active))

    args.output_root.mkdir(parents=True, exist_ok=False)
    novel_path = args.output_root / "stage219-novel-authoritative-color-metadata.csv"
    adverse_path = args.output_root / "stage219-novel-adverse-color-candidates.csv"
    counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    adverse_scene_counts: Counter[str] = Counter()
    adverse_color_counts: Counter[str] = Counter()
    adverse_source_counts: Counter[str] = Counter()

    with inventory.open("r", encoding="utf-8-sig", newline="") as source_handle, \
            novel_path.open("w", encoding="utf-8", newline="") as novel_handle, \
            adverse_path.open("w", encoding="utf-8", newline="") as adverse_handle:
        reader = csv.DictReader(source_handle)
        fields = list(reader.fieldnames or [])
        required = {"key_type", "key", "image_path", "color", "scene", "source_dataset"}
        if missing := required - set(fields):
            raise RuntimeError(f"inventory missing fields: {sorted(missing)}")
        novel_writer = csv.DictWriter(novel_handle, fieldnames=fields)
        adverse_writer = csv.DictWriter(adverse_handle, fieldnames=fields)
        novel_writer.writeheader()
        adverse_writer.writeheader()
        for row in reader:
            counts["inventory_rows"] += 1
            key = (str(row["key_type"]), str(row["key"]))
            if key in active_keys:
                counts["already_in_active_train"] += 1
                continue
            counts["novel_authoritative_color"] += 1
            novel_writer.writerow(row)
            scene = str(row.get("scene") or "unknown")
            color = str(row.get("color") or "unknown")
            sources = [value for value in str(row.get("source_dataset") or "unknown").split("|") if value]
            scene_counts[scene] += 1
            color_counts[color] += 1
            for source in sources:
                source_counts[source] += 1
            if scene in ADVERSE_SCENES:
                counts["novel_adverse_candidate"] += 1
                adverse_writer.writerow(row)
                adverse_scene_counts[scene] += 1
                adverse_color_counts[color] += 1
                for source in sources:
                    adverse_source_counts[source] += 1

    report = {
        "schema_version": "stage219-color-inventory-active-delta-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_metadata_only_delta_ready_for_pixel_and_license_audit",
        "inputs": {
            "stage218_inventory": str(inventory),
            "stage218_inventory_sha256": sha256_file(inventory),
            "active_stage157_manifest": str(active),
            "active_stage157_manifest_sha256": sha256_file(active),
            "active_train_rows": active_train_rows,
            "active_unique_keys": len(active_keys),
        },
        "counts": dict(counts),
        "novel": {
            "scene_counts": dict(scene_counts),
            "color_counts": dict(color_counts),
            "source_counts": dict(source_counts),
        },
        "novel_adverse": {
            "scene_counts": dict(adverse_scene_counts),
            "color_counts": dict(adverse_color_counts),
            "source_counts": dict(adverse_source_counts),
        },
        "outputs": {
            "novel_metadata": str(novel_path),
            "novel_metadata_sha256": sha256_file(novel_path),
            "novel_adverse_candidates": str(adverse_path),
            "novel_adverse_candidates_sha256": sha256_file(adverse_path),
        },
        "policy": {
            "metadata_only": True,
            "image_pixels_opened": 0,
            "validation_rows_used": 0,
            "test_accessed": False,
            "frozen_video_used": False,
            "unknown_promoted_to_night": False,
            "low_luminance_proxy_promoted_to_night": False,
            "training_manifest_created": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage219-color-inventory-active-delta.json"
    atomic_json(report_path, report)
    for path in (novel_path, adverse_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({"status": report["status"], "counts": report["counts"], "novel_adverse": report["novel_adverse"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
