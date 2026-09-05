#!/usr/bin/env python3
"""Audit Stage66 supervision, scene strata, provenance and license eligibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_TRAIN_LICENSES = {
    "CC-BY-4.0",
    "CC-BY-2.0-image / CC-BY-4.0-annotation",
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def supervised(row: dict[str, str], head: str) -> bool:
    value = row.get(f"{head}_supervised", "").strip().lower()
    return True if not value else value in {"1", "true", "yes"}


def summarize(rows: list[dict[str, str]]) -> dict:
    groups = {
        row.get("track_group") or row.get("video_id") or row.get("camera_id")
        for row in rows
    } - {"", None}
    exact_body = [row for row in rows if supervised(row, "body_type") and row.get("body_type") not in {"", "unknown"}]
    exact_color = [row for row in rows if supervised(row, "color") and row.get("color") not in {"", "unknown"}]
    coarse = [row for row in rows if row.get("coarse_body_family") in {"car", "truck"}]
    return {
        "rows": len(rows),
        "unique_image_sha256": len({row.get("sha256") for row in rows if row.get("sha256")}),
        "unique_groups": len(groups),
        "effective_supervision_rows": sum(
            (supervised(row, "body_type") and row.get("body_type") not in {"", "unknown"})
            or (supervised(row, "color") and row.get("color") not in {"", "unknown"})
            or row.get("coarse_body_family") in {"car", "truck"}
            for row in rows
        ),
        "exact_body_supervised_rows": len(exact_body),
        "exact_color_supervised_rows": len(exact_color),
        "coarse_body_family_rows": dict(Counter(row.get("coarse_body_family") for row in coarse)),
        "body_type_counts": dict(Counter(row.get("body_type") for row in exact_body)),
        "color_counts": dict(Counter(row.get("color") for row in exact_color)),
        "scene": {
            "night": sum(truthy(row.get("night")) for row in rows),
            "low_light": sum(truthy(row.get("low_light")) or row.get("lighting") in {"low_light", "night", "low_light_proxy", "night_machine_photometric_consensus"} for row in rows),
            "small": sum(truthy(row.get("small_target")) or row.get("vehicle_size") == "small" for row in rows),
            "occluded": sum(truthy(row.get("occluded")) for row in rows),
            "truncated": sum(truthy(row.get("truncated")) for row in rows),
            "occluded_or_truncated": sum(truthy(row.get("occluded")) or truthy(row.get("truncated")) for row in rows),
            "blurred": sum(truthy(row.get("blur")) for row in rows),
            "weather_counts": dict(Counter(row.get("weather") or "unknown" for row in rows)),
            "lighting_counts": dict(Counter(row.get("lighting") or "unknown" for row in rows)),
            "vehicle_size_counts": dict(Counter(row.get("vehicle_size") or "unknown" for row in rows)),
        },
        "source_dataset_counts": dict(Counter(row.get("source_dataset") or "unknown" for row in rows)),
        "source_license_counts": dict(Counter(row.get("source_license") or "unknown" for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage66 strata evidence")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    marker_rows = sum(
        any(marker in " ".join(str(value).lower() for value in row.values()) for marker in FROZEN_MARKERS)
        for row in rows
    )
    train = [row for row in rows if row.get("split") == "train"]
    invalid_train_license = [
        row for row in train
        if not truthy(row.get("license_train_eligible")) or row.get("source_license") not in ALLOWED_TRAIN_LICENSES
    ]
    by_split = {
        split: summarize([row for row in rows if row.get("split") == split])
        for split in ("train", "validation", "test")
    }
    adverse = [row for row in train if truthy(row.get("stage66_licensed_adverse"))]
    status = "pass" if not marker_rows and not invalid_train_license and len(adverse) == 4033 else "fail"
    report = {
        "schema_version": "attribute-stage66-manifest-strata-audit-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "total_rows": len(rows),
        "by_split": by_split,
        "licensed_adverse": summarize(adverse),
        "invalid_train_license_rows": len(invalid_train_license),
        "frozen_marker_rows": marker_rows,
        "policy": {
            "allowed_train_licenses": sorted(ALLOWED_TRAIN_LICENSES),
            "unknown_is_not_counted_as_exact_supervision": True,
            "coarse_family_truth_is_counted_separately": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "train_effective_supervision_rows": by_split["train"]["effective_supervision_rows"],
        "train_exact_body_rows": by_split["train"]["exact_body_supervised_rows"],
        "train_exact_color_rows": by_split["train"]["exact_color_supervised_rows"],
        "adverse_rows": len(adverse),
    }))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
