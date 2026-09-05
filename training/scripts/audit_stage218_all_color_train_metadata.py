#!/usr/bin/env python3
"""Inventory all train-only exact-color metadata without opening image pixels.

This is a discovery audit, not a training-manifest builder. It deliberately skips
paths that look like test, holdout, evaluation, or frozen-video material and
separates authoritative color truth from model-derived or ambiguous labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TARGET_COLORS = {
    "black", "white", "gray", "silver", "blue", "red", "brown",
    "green", "yellow", "other",
}
SKIP_PATH_MARKERS = {
    "test", "holdout", "future", "evaluation", "eval", "sealed",
    "frozen", "stage148", "stage155", "stage171",
}
FROZEN_ROW_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}
POSITIVE_TRUTH_MARKERS = {
    "exact", "official", "registration", "folder", "ground_truth",
    "ground-truth", "manual", "human", "metadata", "source_label",
    "source-label", "verified",
}
MODEL_TRUTH_MARKERS = {
    "pseudo", "teacher", "consensus", "model", "machine", "weak",
    "coarse", "inferred", "predicted", "prediction",
}
EVIDENCE_FIELDS = (
    "color_supervision", "annotation_source", "review_method",
    "stage65_color_truth_source", "stage66_color_truth_source",
    "taxonomy_v2_color_source", "color_review_status", "review_status",
    "fine_color_recovery", "stage102_supervision",
)
SCENE_FIELDS = (
    "night", "lighting", "weather", "low_light", "night_machine_confidence",
    "scene_lowlight_score", "proposal_source_pool",
)
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def contains_marker(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def should_skip_manifest(path: Path, root: Path) -> tuple[bool, str | None]:
    relative = str(path.relative_to(root)).replace("\\", "/").lower()
    for marker in sorted(SKIP_PATH_MARKERS):
        if marker in relative:
            return True, marker
    return False, None


def classify_color_truth(row: dict[str, str]) -> str:
    color = str(row.get("color") or "").strip().lower()
    if color not in TARGET_COLORS:
        return "not_target_color"
    if not truthy(row.get("color_supervised")):
        return "not_supervised"
    if truthy(row.get("pseudo_label")):
        return "model_derived"
    evidence = " ".join(str(row.get(field) or "") for field in EVIDENCE_FIELDS).lower()
    if contains_marker(evidence, MODEL_TRUTH_MARKERS):
        return "model_derived"
    if contains_marker(evidence, POSITIVE_TRUTH_MARKERS):
        return "authoritative"
    return "ambiguous"


def classify_scene(row: dict[str, str]) -> str:
    night = str(row.get("night") or "").strip().lower()
    lighting = str(row.get("lighting") or "").strip().lower()
    weather = str(row.get("weather") or "").strip().lower()
    low_light = str(row.get("low_light") or "").strip().lower()
    scene_text = " ".join(str(row.get(field) or "") for field in SCENE_FIELDS).lower()
    if "low_luminance_proxy" in lighting or "proxy" in scene_text or "machine" in scene_text:
        return "low_luminance_proxy"
    if truthy(night) or lighting == "night" or weather == "night":
        return "night_metadata_positive"
    if truthy(low_light) or lighting in {"low_light", "low-light", "lowlight"} or weather in {"low_light", "low-light", "lowlight"}:
        return "low_light_metadata_positive"
    if lighting == "daylight" or weather == "daylight":
        return "daylight_metadata_positive"
    return "unknown"


def image_key(row: dict[str, str], manifest: Path) -> tuple[str, str]:
    sha = str(row.get("sha256") or row.get("crop_sha256") or row.get("source_sha256") or "").strip().lower()
    if HEX64.fullmatch(sha):
        return "sha256", sha
    raw = str(row.get("image_path") or "").strip()
    image = Path(raw)
    if not image.is_absolute():
        image = manifest.parent / image
    return "path", os.path.normcase(os.path.normpath(str(image)))


def row_has_frozen_marker(row: dict[str, str]) -> bool:
    searchable = " ".join(
        str(row.get(field) or "")
        for field in ("image_path", "source_manifest", "source_frame_id", "video_id")
    ).lower()
    return contains_marker(searchable, FROZEN_ROW_MARKERS)


def discover(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file() and not path.is_symlink())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-manifests", type=int, default=1000)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"root must be a regular directory: {root}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    candidates = discover(root)
    if len(candidates) > args.max_manifests:
        raise RuntimeError(f"manifest count {len(candidates)} exceeds guard {args.max_manifests}")

    manifest_records: list[dict[str, Any]] = []
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_by_marker: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_scene_counts: Counter[tuple[str, str]] = Counter()

    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    for manifest in candidates:
        skip, marker = should_skip_manifest(manifest, root)
        if skip:
            skipped_by_marker[str(marker)] += 1
            continue
        record: dict[str, Any] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "bytes": manifest.stat().st_size,
            "status": "scanned",
        }
        counts: Counter[str] = Counter()
        try:
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                required = {"image_path", "split", "color"}
                if missing := required - fields:
                    record["status"] = "skipped_missing_fields"
                    record["missing_fields"] = sorted(missing)
                    manifest_records.append(record)
                    continue
                for row_number, row in enumerate(reader, 2):
                    counts["rows_read"] += 1
                    if str(row.get("split") or "").strip().lower() != "train":
                        continue
                    counts["train_rows"] += 1
                    if row_has_frozen_marker(row):
                        raise RuntimeError(f"frozen marker in {manifest}:{row_number}")
                    tier = classify_color_truth(row)
                    counts[f"truth_{tier}"] += 1
                    if tier != "authoritative":
                        continue
                    scene = classify_scene(row)
                    color = str(row.get("color") or "").strip().lower()
                    source = str(row.get("source_dataset") or "unknown").strip() or "unknown"
                    key = image_key(row, manifest)
                    item = unique.setdefault(key, {
                        "key_type": key[0],
                        "key": key[1],
                        "image_path": str(row.get("image_path") or ""),
                        "colors": set(),
                        "scenes": set(),
                        "sources": set(),
                        "track_groups": set(),
                        "manifest_count": 0,
                    })
                    item["colors"].add(color)
                    item["scenes"].add(scene)
                    item["sources"].add(source)
                    track = str(row.get("track_group") or row.get("source_group") or "").strip()
                    if track:
                        item["track_groups"].add(track)
                    item["manifest_count"] += 1
                    source_counts[source] += 1
                    source_scene_counts[(source, scene)] += 1
        except (csv.Error, UnicodeError) as error:
            record["status"] = "failed_parse"
            record["error"] = str(error)
        record["counts"] = dict(counts)
        manifest_records.append(record)

    args.output_root.mkdir(parents=True, exist_ok=False)
    inventory_path = args.output_root / "stage218-authoritative-color-unique-metadata.csv"
    fieldnames = [
        "key_type", "key", "image_path", "color", "scene", "source_dataset",
        "track_group", "manifest_occurrences", "label_conflict", "scene_conflict",
    ]
    unique_scene_counts: Counter[str] = Counter()
    unique_color_counts: Counter[str] = Counter()
    unique_source_counts: Counter[str] = Counter()
    conflicts = 0
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(unique):
            item = unique[key]
            label_conflict = len(item["colors"]) != 1
            scene_conflict = len(item["scenes"]) != 1
            if label_conflict:
                conflicts += 1
                global_counts["unique_label_conflicts_excluded"] += 1
                continue
            color = next(iter(item["colors"]))
            scene = next(iter(item["scenes"])) if not scene_conflict else "conflicting_scene_metadata"
            sources = sorted(item["sources"])
            tracks = sorted(item["track_groups"])
            writer.writerow({
                "key_type": item["key_type"],
                "key": item["key"],
                "image_path": item["image_path"],
                "color": color,
                "scene": scene,
                "source_dataset": "|".join(sources),
                "track_group": "|".join(tracks),
                "manifest_occurrences": item["manifest_count"],
                "label_conflict": "false",
                "scene_conflict": str(scene_conflict).lower(),
            })
            global_counts["unique_authoritative_color_images"] += 1
            unique_scene_counts[scene] += 1
            unique_color_counts[color] += 1
            for source in sources:
                unique_source_counts[source] += 1

    scanned = [r for r in manifest_records if r["status"] == "scanned"]
    report = {
        "schema_version": "stage218-all-color-train-metadata-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_metadata_only_candidate_inventory",
        "root": str(root),
        "discovery": {
            "csv_files_found": len(candidates),
            "manifests_scanned": len(scanned),
            "manifests_skipped_by_path_marker": sum(skipped_by_marker.values()),
            "skip_marker_counts": dict(skipped_by_marker),
            "manifests_skipped_or_failed_for_other_reasons": len(manifest_records) - len(scanned),
        },
        "unique": {
            **dict(global_counts),
            "scene_counts": dict(unique_scene_counts),
            "color_counts": dict(unique_color_counts),
            "source_counts": dict(unique_source_counts),
            "authoritative_label_conflicts_excluded": conflicts,
        },
        "raw_authoritative_occurrences": {
            "source_counts": dict(source_counts),
            "source_scene_counts": {f"{source}|{scene}": count for (source, scene), count in source_scene_counts.items()},
        },
        "manifests": manifest_records,
        "outputs": {
            "unique_metadata_csv": str(inventory_path),
            "unique_metadata_csv_sha256": sha256_file(inventory_path),
        },
        "policy": {
            "metadata_only": True,
            "image_pixels_opened": 0,
            "validation_or_test_rows_used": 0,
            "test_path_markers_skipped": True,
            "frozen_video_used": False,
            "unknown_scene_promoted_to_night": False,
            "low_luminance_proxy_promoted_to_night": False,
            "model_derived_color_promoted_to_truth": False,
            "training_manifest_created": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage218-all-color-train-metadata-audit.json"
    atomic_json(report_path, report)
    for path in (inventory_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({
        "status": report["status"],
        "discovery": report["discovery"],
        "unique": report["unique"],
        "outputs": report["outputs"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
