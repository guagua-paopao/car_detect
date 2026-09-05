#!/usr/bin/env python3
"""Build immutable test-only views after a passing Stage106 component gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_v2_decoupled_shared_test_once import (  # noqa: E402
    FROZEN_MARKERS,
    load_stage106_parameters,
    require_sha,
    sha256,
    truthy,
)


def sample_digest(row: dict[str, str]) -> str:
    for field in ("crop_sha256", "sha256", "source_image_sha256"):
        value = str(row.get(field, "")).strip().lower()
        if value:
            return value
    return ""


def build_view(
    name: str,
    source: Path,
    output: Path,
    safety_root: Path,
) -> tuple[dict[str, Any], set[str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{name} source manifest has no header")
        fields = list(reader.fieldnames)
        source_rows = list(reader)
    selected: list[dict[str, str]] = []
    digests: set[str] = set()
    labels: Counter[str] = Counter()
    groups: set[str] = set()
    cameras: set[str] = set()
    videos: set[str] = set()
    for line, raw in enumerate(source_rows, start=2):
        if str(raw.get("split", "")).strip().lower() != "test":
            continue
        row = dict(raw)
        searchable = " ".join(str(value).lower() for value in row.values())
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"{name} frozen marker at source line {line}")
        if str(row.get("review_status", "")).strip().lower() != "approved":
            raise RuntimeError(f"{name} unapproved test row at source line {line}")
        raw_image = Path(row.get("image_path", ""))
        resolved = (
            raw_image.resolve()
            if raw_image.is_absolute()
            else (source.parent / raw_image).resolve()
        )
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"{name} test image escapes safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["image_path"] = os.path.relpath(resolved, output.parent).replace("\\", "/")
        row["split"] = "test"
        selected.append(row)
        digest = sample_digest(row)
        if digest:
            digests.add(digest)
        if truthy(row.get("body_type_supervised")):
            labels[f"body:{row.get('body_type', 'unknown')}"] += 1
        if truthy(row.get("color_supervised")):
            labels[f"color:{row.get('color', 'unknown')}"] += 1
        group = row.get("track_key") or row.get("track_id") or row.get("source_group")
        if group:
            groups.add(str(group))
        camera = row.get("camera_id") or row.get("camera")
        if camera:
            cameras.add(str(camera))
        video = row.get("source_video") or row.get("video_id")
        if video:
            videos.add(str(video))
    if not selected:
        raise RuntimeError(f"{name} source has no test rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return (
        {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "source_rows": len(source_rows),
            "test_rows": len(selected),
            "test_groups": len(groups),
            "test_cameras": len(cameras),
            "test_videos": len(videos),
            "label_support": dict(sorted(labels.items())),
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
            "split_values": ["test"],
            "labels_modified": False,
            "membership_reassigned": False,
        },
        digests,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage106-state", type=Path, required=True)
    parser.add_argument("--expected-stage106-state-sha256", required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--expected-hard-manifest-sha256", required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--expected-vfg-manifest-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite Stage107 test views: {args.output_root}")
    # This is intentionally the first operation that may authorize reading
    # source test manifests.  A non-passing Stage106 state fails beforehand.
    parameters = load_stage106_parameters(
        args.stage106_state, args.expected_stage106_state_sha256
    )
    pinned = (
        (args.hard_manifest, args.expected_hard_manifest_sha256, "hard manifest"),
        (args.ua_manifest, args.expected_ua_manifest_sha256, "UA manifest"),
        (args.vfg_manifest, args.expected_vfg_manifest_sha256, "VFG manifest"),
    )
    for path, expected, label in pinned:
        require_sha(path, expected, label)
    safety_root = args.datasets_safety_root.resolve()
    if not safety_root.is_dir():
        raise RuntimeError("datasets safety root is missing")
    args.output_root.mkdir(parents=True, exist_ok=False)
    sources = {
        "hard": args.hard_manifest,
        "ua": args.ua_manifest,
        "vfg": args.vfg_manifest,
    }
    reports: dict[str, Any] = {}
    digest_sets: dict[str, set[str]] = {}
    for name, source in sources.items():
        reports[name], digest_sets[name] = build_view(
            name,
            source.resolve(),
            args.output_root / f"{name}.test.csv",
            safety_root,
        )
    intersections: dict[str, int] = {}
    names = sorted(digest_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            intersections[f"{left}:{right}"] = len(digest_sets[left] & digest_sets[right])
    report = {
        "schema_version": "stage107-test-views-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_test_views_ready",
        "stage106_state": str(args.stage106_state.resolve()),
        "stage106_state_sha256": sha256(args.stage106_state),
        "fixed_validation_parameters": parameters,
        "views": reports,
        "cross_view_exact_digest_intersections": intersections,
        "policy": {
            "source_split_membership_preserved": True,
            "only_preexisting_test_rows_selected": True,
            "labels_modified": False,
            "test_accessed": True,
            "test_used_for_selection": False,
            "threshold_or_temperature_search": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage107-test-views-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_path.with_suffix(report_path.suffix + ".sha256")).write_text(
        f"{sha256(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
