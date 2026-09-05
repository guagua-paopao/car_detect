#!/usr/bin/env python3
"""Append an audited train-only color supplement while freezing val/test rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def permissive_training_license(row: dict[str, str]) -> bool:
    """Conservatively reject base rows carrying research/non-commercial terms."""
    if not truthy(row.get("license_train_eligible")):
        return False
    license_text = str(row.get("source_license", "")).strip().lower()
    restricted_markers = (
        "research", "education-only", "education only", "non-commercial",
        "noncommercial", "non commercial", "nc-only",
    )
    return bool(license_text) and not any(marker in license_text for marker in restricted_markers)


def exact_rows_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row.get(field, "") for field in fields], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_dataset_path(dataset_root: Path, relative: str) -> Path | None:
    value = Path(relative)
    if value.is_absolute():
        return None
    root = dataset_root.resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--base-license-policy",
        choices=("all", "permissive"),
        default="all",
        help="Use all eligible base color rows or exclude research/non-commercial base training sources.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage48 manifest evidence")
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(args.dataset_root)

    base_fields, base_rows = read_csv(args.base)
    supplement_fields, supplement_rows = read_csv(args.supplement)
    fields = base_fields + [field for field in supplement_fields if field not in base_fields]
    baseline_eval = [dict(row) for row in base_rows if row.get("split") in {"validation", "test"}]
    baseline_eval_digest = exact_rows_digest(baseline_eval, base_fields)
    retained_base = [
        row for row in base_rows
        if row.get("split") in {"validation", "test"}
        or (
            row.get("split") == "train"
            and truthy(row.get("color_supervised"))
            and row.get("color") not in {"", "unknown"}
            and (
                args.base_license_policy == "all"
                or permissive_training_license(row)
            )
        )
    ]
    base_color_rows = [
        row for row in base_rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and row.get("color") not in {"", "unknown"}
    ]
    seen_paths = {row.get("image_path", "") for row in base_rows}
    seen_hashes = {row.get("sha256", "").lower() for row in base_rows if row.get("sha256")}
    accepted: list[dict[str, str]] = []
    rejected = Counter()

    for source in supplement_rows:
        if source.get("split") != "train":
            rejected["not_train"] += 1
            continue
        if source.get("source_license") != "CC0-1.0" or not truthy(source.get("license_train_eligible")):
            rejected["license_not_train_eligible"] += 1
            continue
        if not truthy(source.get("formal_train_eligible")):
            rejected["formal_train_ineligible"] += 1
            continue
        if not truthy(source.get("color_supervised")) or source.get("color") in {"", "unknown"}:
            rejected["missing_color_supervision"] += 1
            continue
        if truthy(source.get("body_type_supervised")) or source.get("body_type") not in {"", "unknown"}:
            rejected["unexpected_body_supervision"] += 1
            continue
        relative = source.get("image_path", "")
        image_path = safe_dataset_path(args.dataset_root, relative)
        if image_path is None or not image_path.is_file():
            rejected["unsafe_or_missing_crop"] += 1
            continue
        digest = sha256(image_path)
        declared_digest = (source.get("sha256") or source.get("crop_sha256") or "").lower()
        if not declared_digest or digest.lower() != declared_digest:
            rejected["crop_sha256_mismatch"] += 1
            continue
        if relative in seen_paths:
            rejected["duplicate_path"] += 1
            continue
        if digest.lower() in seen_hashes:
            rejected["duplicate_sha256"] += 1
            continue
        row = {field: source.get(field, "") for field in fields}
        accepted.append(row)
        seen_paths.add(relative)
        seen_hashes.add(digest.lower())

    output_rows = retained_base + accepted
    output_eval = [row for row in output_rows if row.get("split") in {"validation", "test"}]
    output_eval_digest = exact_rows_digest(output_eval, base_fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    policy = {
        "frozen_video_not_used": True,
        "test_not_loaded_or_used_for_selection": True,
        "production_model_unchanged": True,
        "deployment_not_performed": True,
        "supplement_train_only": bool(accepted) and all(row.get("split") == "train" for row in accepted),
        "supplement_color_only": bool(accepted) and all(
            truthy(row.get("color_supervised")) and not truthy(row.get("body_type_supervised")) for row in accepted
        ),
        "supplement_cc0_only": bool(accepted) and all(row.get("source_license") == "CC0-1.0" for row in accepted),
        "training_rows_color_supervised_only": all(
            truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown"}
            for row in output_rows if row.get("split") == "train"
        ),
        "validation_test_rows_exactly_unchanged": (
            len(output_eval) == len(baseline_eval) and output_eval_digest == baseline_eval_digest
        ),
    }
    report = {
        "schema_version": "stage48-real-color-mixture-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(policy.values()) and len(accepted) >= 100 else "fail",
        "policy": policy,
        "base_manifest": str(args.base.resolve()),
        "base_manifest_sha256": sha256(args.base),
        "supplement_manifest": str(args.supplement.resolve()),
        "supplement_manifest_sha256": sha256(args.supplement),
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256(args.output),
        "base_rows": len(base_rows),
        "base_license_policy": args.base_license_policy,
        "base_train_color_rows_before_license_filter": len(base_color_rows),
        "base_rows_retained": len(retained_base),
        "base_train_color_rows_retained": sum(row.get("split") == "train" for row in retained_base),
        "base_train_color_rows_excluded_by_license_policy": (
            len(base_color_rows) - sum(row.get("split") == "train" for row in retained_base)
        ),
        "retained_base_train_license_counts": dict(sorted(Counter(
            row.get("source_license", "unknown")
            for row in retained_base if row.get("split") == "train"
        ).items())),
        "base_train_noncolor_rows_omitted": len(base_rows) - len(retained_base),
        "supplement_input_rows": len(supplement_rows),
        "supplement_accepted_rows": len(accepted),
        "output_rows": len(output_rows),
        "rejected": dict(sorted(rejected.items())),
        "accepted_color_counts": dict(sorted(Counter(row.get("color", "unknown") for row in accepted).items())),
        "accepted_lighting_counts": dict(sorted(Counter(row.get("lighting", "unknown") for row in accepted).items())),
        "accepted_size_counts": dict(sorted(Counter(row.get("vehicle_size", "unknown") for row in accepted).items())),
        "baseline_eval_rows": len(baseline_eval),
        "baseline_eval_exact_digest": baseline_eval_digest,
        "output_eval_exact_digest": output_eval_digest,
        "decision": "Stage48 color-specialist training may start only if status=pass; use --skip-test and body loss 0 during candidate iteration",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
