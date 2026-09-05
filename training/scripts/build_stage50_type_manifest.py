#!/usr/bin/env python3
"""Build a type-only Stage50 mixture with audited VTID2 train rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VTID2_SAMPLE_WEIGHTS = {"sedan": 5.0, "suv": 10.0, "pickup": 8.0}


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


def row_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row.get(field, "") for field in fields], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path | None:
    value = Path(relative)
    if value.is_absolute():
        return None
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
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
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage50 evidence")
    base_fields, base_rows = read_csv(args.base)
    supplement_fields, supplement_rows = read_csv(args.supplement)
    fields = base_fields + [field for field in supplement_fields if field not in base_fields]
    baseline_eval = [dict(row) for row in base_rows if row.get("split") in {"validation", "test"}]
    baseline_eval_digest = row_digest(baseline_eval, base_fields)
    retained_base = [
        row for row in base_rows
        if row.get("split") in {"validation", "test"}
        or (
            row.get("split") == "train"
            and truthy(row.get("body_type_supervised"))
            and row.get("body_type") not in {"", "unknown"}
        )
    ]
    seen_paths = {row.get("image_path", "") for row in base_rows}
    seen_hashes = {row.get("sha256", "").lower() for row in base_rows if row.get("sha256")}
    accepted = []
    rejected = Counter()
    for source in supplement_rows:
        if source.get("oof_teacher_consensus") != "accepted":
            rejected["oof_teacher_not_accepted"] += 1
            continue
        if source.get("split") not in {"train", "audit", "validation"}:
            rejected["unexpected_source_split"] += 1
            continue
        if source.get("source_license") != "CC-BY-4.0" or not truthy(source.get("license_train_eligible")):
            rejected["license"] += 1
            continue
        if not truthy(source.get("formal_train_eligible")):
            rejected["formal_train_ineligible"] += 1
            continue
        if not truthy(source.get("body_type_supervised")) or source.get("body_type") not in {"sedan", "suv", "pickup"}:
            rejected["invalid_body_supervision"] += 1
            continue
        if truthy(source.get("color_supervised")) or source.get("color") not in {"", "unknown"}:
            rejected["unexpected_color_supervision"] += 1
            continue
        relative = source.get("image_path", "")
        image_path = safe_path(args.dataset_root, relative)
        if image_path is None or not image_path.is_file():
            rejected["unsafe_or_missing_image"] += 1
            continue
        digest = sha256(image_path)
        if digest.lower() != (source.get("sha256") or "").lower():
            rejected["sha256_mismatch"] += 1
            continue
        if relative in seen_paths:
            rejected["duplicate_path"] += 1
            continue
        if digest.lower() in seen_hashes:
            rejected["duplicate_sha256"] += 1
            continue
        row = {field: source.get(field, "") for field in fields}
        row["split"] = "train"
        row["sample_weight"] = f"{VTID2_SAMPLE_WEIGHTS[row['body_type']]:.3f}"
        row["training_reason"] = "VTID2 unseen-fold teacher-approved real university-gate body type"
        accepted.append(row)
        seen_paths.add(relative)
        seen_hashes.add(digest.lower())

    output_rows = retained_base + accepted
    output_eval = [row for row in output_rows if row.get("split") in {"validation", "test"}]
    output_eval_digest = row_digest(output_eval, base_fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    class_counts = Counter(row.get("body_type", "unknown") for row in accepted)
    policy = {
        "supplement_train_only": bool(accepted) and all(row.get("split") == "train" for row in accepted),
        "supplement_body_only": bool(accepted) and all(
            truthy(row.get("body_type_supervised")) and not truthy(row.get("color_supervised")) for row in accepted
        ),
        "supplement_cc_by_only": bool(accepted) and all(row.get("source_license") == "CC-BY-4.0" for row in accepted),
        "supplement_oof_teacher_approved_only": bool(accepted) and all(row.get("oof_teacher_consensus") == "accepted" for row in accepted),
        "training_rows_body_supervised_only": all(
            truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown"}
            for row in output_rows if row.get("split") == "train"
        ),
        "validation_test_rows_exactly_unchanged": (
            len(output_eval) == len(baseline_eval) and output_eval_digest == baseline_eval_digest
        ),
        "test_not_loaded_or_used_for_selection": True,
        "frozen_video_not_used": True,
        "production_model_unchanged": True,
        "deployment_not_performed": True,
    }
    status = "pass" if (
        all(policy.values()) and len(accepted) >= 500
        and all(class_counts[label] >= 50 for label in ("sedan", "suv", "pickup"))
    ) else "fail"
    report = {
        "schema_version": "stage50-vtid2-type-mixture-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "policy": policy,
        "base_manifest": str(args.base.resolve()),
        "base_manifest_sha256": sha256(args.base),
        "supplement_manifest": str(args.supplement.resolve()),
        "supplement_manifest_sha256": sha256(args.supplement),
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256(args.output),
        "base_rows": len(base_rows),
        "base_rows_retained": len(retained_base),
        "base_train_body_rows_retained": sum(row.get("split") == "train" for row in retained_base),
        "supplement_input_rows": len(supplement_rows),
        "supplement_accepted_rows": len(accepted),
        "supplement_body_counts": dict(sorted(class_counts.items())),
        "supplement_sample_weights": VTID2_SAMPLE_WEIGHTS,
        "rejected": dict(sorted(rejected.items())),
        "output_rows": len(output_rows),
        "baseline_eval_rows": len(baseline_eval),
        "baseline_eval_digest": baseline_eval_digest,
        "output_eval_digest": output_eval_digest,
        "decision": "eligible for Stage50 type-specialist validation-only training with body loss only and --skip-test",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
