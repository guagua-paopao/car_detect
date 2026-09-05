#!/usr/bin/env python3
"""Inspect Stage177 unknown train rows without relabeling or touching held-out data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48")
NUMBER_RUN = re.compile(r"\d+")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_template(path_text: str) -> str:
    path = Path(path_text)
    stem = NUMBER_RUN.sub("#", path.stem.casefold())
    return path.parent.as_posix().casefold() + "/" + stem + path.suffix.casefold()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(marker in str(value).lower() for value in (args.manifest, args.output) for marker in FROZEN_MARKERS):
        raise ValueError("frozen-video path is forbidden")
    actual_sha = sha256(args.manifest)
    if actual_sha.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_sha}")

    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    templates: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "image_path", "source_dataset", "stage177_scene_label"}
        if missing := required - set(reader.fieldnames or []):
            raise RuntimeError(f"missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("split") != "train" or row.get("stage177_scene_label") != "unknown":
                continue
            path_text = str(row.get("image_path") or "")
            if any(marker in path_text.lower() for marker in FROZEN_MARKERS):
                raise RuntimeError("frozen marker found in manifest")
            source = str(row.get("source_dataset") or "unknown")
            counts["unknown_rows"] += 1
            counts["dedup_eligible_rows"] += int(truthy(row.get("stage177_dedup_eligible")))
            counts["effective_representatives"] += int(truthy(row.get("stage177_effective_representative")))
            stat = by_source[source]
            stat["rows"] += 1
            stat["dedup_eligible"] += int(truthy(row.get("stage177_dedup_eligible")))
            stat["effective_representatives"] += int(truthy(row.get("stage177_effective_representative")))
            stat["camera_id"] += int(bool(str(row.get("camera_id") or "").strip()))
            stat["video_id"] += int(bool(str(row.get("video_id") or "").strip()))
            stat["track_group"] += int(bool(str(row.get("track_group") or "").strip()))
            stat["source_group"] += int(bool(str(row.get("source_group") or "").strip()))
            stat["source_frame_id"] += int(bool(str(row.get("source_frame_id") or "").strip()))
            stat["scene_group_key"] += int(bool(str(row.get("stage177_scene_group_key") or "").strip()))
            stat["machine_night_candidate"] += int(truthy(row.get("stage177_machine_night_candidate")))
            templates[source][path_template(path_text)] += 1
            if len(examples[source]) < 5:
                examples[source].append(path_text)

    report = {
        "schema_version": "stage197-unknown-scene-scope-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "inspection_complete_no_relabeling",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": actual_sha,
        "counts": dict(counts),
        "per_source": {
            source: {
                **dict(stat),
                "top_path_templates": templates[source].most_common(10),
                "example_paths": examples[source],
            }
            for source, stat in sorted(by_source.items())
        },
        "policy": {
            "pixels_opened": False,
            "labels_modified": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
