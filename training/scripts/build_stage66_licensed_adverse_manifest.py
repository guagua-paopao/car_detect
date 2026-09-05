#!/usr/bin/env python3
"""Build a production-eligible v1 manifest with licensed adverse partial labels."""

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
V1_BODY = {
    "sedan", "suv", "mpv", "van", "pickup", "bus",
    "light_truck", "heavy_truck", "other", "unknown",
}
V1_COLORS = {
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown",
}
COLOR_MAP = {
    "gray": "silver_gray",
    "silver": "silver_gray",
    "yellow": "yellow_orange",
    "brown": "brown_beige",
}


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def evaluation_digest(rows: list[dict[str, str]]) -> str:
    payload = [
        (row.get("split", ""), row.get("sha256", ""), row.get("image_path", ""))
        for row in rows if row.get("split") in {"validation", "test"}
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def licensed_train_row(row: dict[str, str]) -> bool:
    return truthy(row.get("license_train_eligible")) and row.get("source_license", "") in ALLOWED_TRAIN_LICENSES


def v1_adverse_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    body = row.get("body_type", "unknown").strip().lower() or "unknown"
    if body == "truck":
        row["body_type"] = "unknown"
        row["body_type_supervised"] = "false"
        row["coarse_body_family"] = "truck"
        row["stage66_body_truth_source"] = "official_coarse_truck_family"
    elif body not in V1_BODY:
        row["body_type"] = "unknown"
        row["body_type_supervised"] = "false"
        row["stage66_body_truth_source"] = "unsupported_v1_label_fail_closed"
    color = COLOR_MAP.get(row.get("color", "").strip().lower(), row.get("color", "").strip().lower())
    if color not in V1_COLORS:
        row["color"] = "unknown"
        row["color_supervised"] = "false"
        row["stage66_color_truth_source"] = "unsupported_v1_label_fail_closed"
    else:
        row["color"] = color
    row["stage66_licensed_adverse"] = "true"
    row["training_reason"] = "stage66_licensed_real_adverse_partial"
    return row


def build(base_rows: list[dict[str, str]], adverse_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    retained: list[dict[str, str]] = []
    rejected_train = Counter()
    for row in base_rows:
        if row.get("split") != "train":
            retained.append(dict(row))
        elif licensed_train_row(row):
            retained.append(dict(row))
        else:
            rejected_train[row.get("source_license", "<missing>") or "<missing>"] += 1
    existing_sha = {row.get("sha256") for row in retained if row.get("sha256")}
    appended: list[dict[str, str]] = []
    rejection = Counter()
    for source in adverse_rows:
        if not truthy(source.get("adverse_supervised")):
            continue
        if source.get("split") != "train":
            rejection["not_train"] += 1
            continue
        if not licensed_train_row(source):
            rejection["license_not_allowed"] += 1
            continue
        digest = source.get("sha256", "")
        if not digest:
            rejection["missing_sha256"] += 1
            continue
        if digest in existing_sha:
            rejection["duplicate_sha256"] += 1
            continue
        row = v1_adverse_row(source)
        appended.append(row)
        existing_sha.add(digest)
    output = retained + appended
    source_eval = evaluation_digest(base_rows)
    output_eval = evaluation_digest(output)
    if source_eval != output_eval:
        raise RuntimeError("validation/test membership changed")
    train_groups = {
        row.get("track_group") or row.get("video_id") or row.get("camera_id")
        for row in output if row.get("split") == "train"
    } - {"", None}
    heldout_groups = {
        row.get("track_group") or row.get("video_id") or row.get("camera_id")
        for row in output if row.get("split") in {"validation", "test"}
    } - {"", None}
    overlap = train_groups & heldout_groups
    if overlap:
        raise RuntimeError(f"source-group leakage detected: {len(overlap)} groups")
    appended_counts = {
        "rows": len(appended),
        "night": sum(truthy(row.get("night")) for row in appended),
        "small": sum(row.get("vehicle_size", "").lower() == "small" or truthy(row.get("small_target")) for row in appended),
        "occluded_or_truncated": sum(truthy(row.get("occluded")) or truthy(row.get("truncated")) for row in appended),
        "coarse_car": sum(row.get("coarse_body_family") == "car" for row in appended),
        "coarse_truck": sum(row.get("coarse_body_family") == "truck" for row in appended),
        "exact_body": sum(truthy(row.get("body_type_supervised")) and row.get("body_type") != "unknown" for row in appended),
        "exact_color": sum(truthy(row.get("color_supervised")) and row.get("color") != "unknown" for row in appended),
    }
    audit = {
        "base_rows": len(base_rows),
        "retained_base_rows": len(retained),
        "rejected_base_train_by_license": dict(rejected_train),
        "appended": appended_counts,
        "append_rejections": dict(rejection),
        "output_rows": len(output),
        "output_split_counts": dict(Counter(row.get("split", "") for row in output)),
        "evaluation_membership_preserved": True,
        "evaluation_digest": source_eval,
        "cross_split_group_overlap": 0,
        "allowed_train_licenses": sorted(ALLOWED_TRAIN_LICENSES),
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--adverse-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage66 manifest evidence")
    base_fields, base_rows = read_rows(args.base_manifest)
    adverse_fields, adverse_rows = read_rows(args.adverse_manifest)
    output, audit = build(base_rows, adverse_rows)
    fields = list(dict.fromkeys([*base_fields, *adverse_fields, "stage66_body_truth_source", "stage66_color_truth_source", "stage66_licensed_adverse"]))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    report = {
        "schema_version": "attribute-stage66-licensed-adverse-manifest-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "adverse_manifest": str(args.adverse_manifest.resolve()),
        "adverse_manifest_sha256": sha256(args.adverse_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        **audit,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
