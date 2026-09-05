#!/usr/bin/env python3
"""Validate a strict train-only supplement appended to a hash-locked legacy manifest.

The historical base manifest predates the current metadata enums.  This validator
does not silently rewrite those rows: it requires every base field and evaluation
row to be byte-for-byte equivalent at CSV-field level, then applies the current
strict contract to every appended row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BOOLS = {"true", "false"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def rows_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {field: row.get(field, "") for field in fields}
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-supplement", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation evidence: {args.output}")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    body_types = set(labels["body_types"])
    colors = set(labels["colors"])
    qualities = set(labels["crop_qualities"])
    viewpoints = set(labels["viewpoints"])
    base_fields, base_rows = load_csv(args.base_manifest)
    merged_fields, merged_rows = load_csv(args.merged_manifest)
    errors: list[str] = []
    warnings: list[str] = []

    if len(merged_rows) < len(base_rows):
        errors.append("merged manifest is shorter than locked base")
    missing_base_fields = sorted(set(base_fields) - set(merged_fields))
    if missing_base_fields:
        errors.append(f"merged manifest dropped base fields: {missing_base_fields}")

    prefix_mismatches = 0
    for index, base in enumerate(base_rows):
        if index >= len(merged_rows):
            break
        merged = merged_rows[index]
        if any(base.get(field, "") != merged.get(field, "") for field in base_fields):
            prefix_mismatches += 1
            if prefix_mismatches <= 10:
                errors.append(f"locked base row changed at CSV row {index + 2}")
    if prefix_mismatches > 10:
        errors.append(f"additional locked base row mismatches: {prefix_mismatches - 10}")

    base_eval = [row for row in base_rows if row.get("split") in {"validation", "test"}]
    merged_eval = [row for row in merged_rows if row.get("split") in {"validation", "test"}]
    base_eval_digest = rows_digest(base_eval, base_fields)
    merged_eval_digest = rows_digest(merged_eval, base_fields)
    if len(base_eval) != len(merged_eval) or base_eval_digest != merged_eval_digest:
        errors.append("validation/test rows or fields changed")

    supplement = merged_rows[len(base_rows):]
    if len(supplement) < args.minimum_supplement:
        errors.append(
            f"supplement row count {len(supplement)} is below {args.minimum_supplement}"
        )

    all_paths: set[str] = set()
    base_dhash = {row.get("dhash64", "") for row in base_rows if len(row.get("dhash64", "")) == 16}
    supplement_dhash: set[str] = set()
    base_groups = {
        value
        for row in base_rows
        for value in (row.get("camera_id", ""), row.get("video_id", ""), row.get("track_group", ""))
        if value
    }
    supplement_counts: Counter[str] = Counter()
    for row_number, row in enumerate(merged_rows, start=2):
        image_path = row.get("image_path", "").replace("\\", "/")
        if not image_path:
            errors.append(f"row {row_number}: empty image_path")
        elif image_path in all_paths:
            errors.append(f"row {row_number}: duplicate image_path {image_path}")
        all_paths.add(image_path)

    for offset, row in enumerate(supplement, start=len(base_rows) + 2):
        prefix = f"row {offset}"
        image_path = row.get("image_path", "").replace("\\", "/")
        if image_path and not (args.dataset_root / image_path).is_file():
            errors.append(f"{prefix}: missing image {image_path}")
        if row.get("split") != "train":
            errors.append(f"{prefix}: supplement must be train-only")
        if row.get("review_status") != "approved":
            errors.append(f"{prefix}: supplement is not approved")
        if row.get("body_type") not in body_types:
            errors.append(f"{prefix}: invalid body_type {row.get('body_type')!r}")
        if row.get("color") not in colors:
            errors.append(f"{prefix}: invalid color {row.get('color')!r}")
        if row.get("crop_quality") not in qualities:
            errors.append(f"{prefix}: invalid crop_quality {row.get('crop_quality')!r}")
        if row.get("viewpoint") not in viewpoints:
            errors.append(f"{prefix}: invalid viewpoint {row.get('viewpoint')!r}")
        for field in ("blur", "occluded", "truncated", "night"):
            if row.get(field, "").strip().lower() not in BOOLS:
                errors.append(f"{prefix}: {field} must be true or false")
        for head in ("body_type", "color"):
            supervised = row.get(f"{head}_supervised", "").strip().lower()
            if supervised not in BOOLS:
                errors.append(f"{prefix}: {head}_supervised must be true or false")
            elif supervised == "false" and row.get(head) != "unknown":
                errors.append(f"{prefix}: unsupervised {head} must be unknown")
        if row.get("formal_train_eligible", "").strip().lower() != "true":
            errors.append(f"{prefix}: formal_train_eligible must be true")
        if row.get("license_train_eligible", "").strip().lower() != "true":
            errors.append(f"{prefix}: license_train_eligible must be true")
        groups = {
            row.get("camera_id", ""), row.get("video_id", ""), row.get("track_group", "")
        } - {""}
        if groups & base_groups:
            errors.append(f"{prefix}: source group overlaps locked base")
        dhash = row.get("dhash64", "")
        if len(dhash) != 16:
            errors.append(f"{prefix}: invalid dhash64")
        elif dhash in base_dhash or dhash in supplement_dhash:
            errors.append(f"{prefix}: exact dHash overlap")
        supplement_dhash.add(dhash)
        supplement_counts["body_supervised"] += is_true(row.get("body_type_supervised", ""))
        supplement_counts["color_supervised"] += is_true(row.get("color_supervised", ""))
        supplement_counts["small"] += is_true(row.get("small_target", ""))
        supplement_counts["occluded"] += is_true(row.get("occluded", ""))
        supplement_counts["truncated"] += is_true(row.get("truncated", ""))

    # Legacy metadata debt is evidence, not silently converted into false facts.
    legacy_metadata = Counter()
    for row in base_rows:
        if row.get("crop_quality") not in qualities:
            legacy_metadata["invalid_crop_quality"] += 1
        if row.get("viewpoint") not in viewpoints:
            legacy_metadata["invalid_viewpoint"] += 1
        for field in ("blur", "occluded", "truncated", "night"):
            if row.get(field, "").strip().lower() not in BOOLS:
                legacy_metadata[f"non_boolean_{field}"] += 1
    if legacy_metadata:
        warnings.append(
            "locked Stage50 rows contain legacy/unknown metadata; preserved exactly and excluded "
            "from supplement-only quota claims"
        )

    report = {
        "schema_version": "locked-base-supplement-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "merged_manifest": str(args.merged_manifest.resolve()),
        "merged_manifest_sha256": sha256(args.merged_manifest),
        "dataset_root": str(args.dataset_root.resolve()),
        "base_rows": len(base_rows),
        "merged_rows": len(merged_rows),
        "supplement_rows": len(supplement),
        "locked_base_prefix_exact": prefix_mismatches == 0,
        "evaluation_rows_unchanged": len(base_eval) == len(merged_eval) and base_eval_digest == merged_eval_digest,
        "base_evaluation_digest": base_eval_digest,
        "merged_evaluation_digest": merged_eval_digest,
        "supplement_counts": dict(sorted(supplement_counts.items())),
        "legacy_base_metadata_debt": dict(sorted(legacy_metadata.items())),
        "warnings": warnings,
        "errors": errors,
        "policy": {
            "legacy_base_hash_locked_and_preserved": True,
            "strict_contract_applied_to_supplement": True,
            "supplement_train_only": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement),
        "errors": len(errors),
        "warnings": len(warnings),
        "evaluation_rows_unchanged": report["evaluation_rows_unchanged"],
    }, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
