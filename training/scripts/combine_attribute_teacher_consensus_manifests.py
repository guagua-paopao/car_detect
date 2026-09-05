#!/usr/bin/env python3
"""Combine isolated strict teacher-consensus manifests without weakening their gates."""

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


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_policy(head: str, policy: dict) -> None:
    if head == "body":
        required = {
            "all_teachers_exact_agreement": True,
            "all_views_exact_agreement": True,
            "official_coarse_class_compatibility_required": True,
            "frozen_video_used": False,
        }
    else:
        required = {
            "foreground_or_official_labels_retained_only_not_replaced": True,
            "rejected_rows_changed_to_unknown_unsupervised": True,
            "test_split_not_used": True,
            "frozen_video_not_used": True,
        }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"{head} teacher policy mismatch for {key}")


def accepted(head: str, row: dict[str, str]) -> bool:
    if row.get("split") != "train" or not truthy(row.get("formal_train_eligible")):
        return False
    if head == "body":
        return (
            row.get("teacher_consensus") == "accepted"
            and truthy(row.get("body_type_supervised"))
            and row.get("body_type") not in {"", "unknown"}
        )
    return (
        row.get("color_teacher_consensus") == "accepted"
        and truthy(row.get("color_supervised"))
        and row.get("color") not in {"", "unknown"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("body", "color"), required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.manifest) == len(args.report) == len(args.pool)):
        raise ValueError("--manifest, --report and --pool counts must match")
    if len(set(args.pool)) != len(args.pool):
        raise ValueError("teacher source pool names must be unique")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite combined teacher evidence")

    fields: list[str] = []
    combined: dict[str, dict[str, str]] = {}
    source_evidence = []
    pool_counts = Counter()
    label_counts = Counter()
    conflicts = Counter()
    rejected_paths: set[str] = set()
    for manifest_path, report_path, pool in zip(args.manifest, args.report, args.pool):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("output_manifest_sha256") != sha256(manifest_path):
            raise RuntimeError(f"teacher manifest hash mismatch for pool {pool}")
        validate_policy(args.head, report.get("policy", {}))
        manifest_fields, rows = load(manifest_path)
        for field in manifest_fields:
            if field not in fields:
                fields.append(field)
        accepted_rows = [row for row in rows if accepted(args.head, row)]
        if len(accepted_rows) != int(report.get("accepted_rows", -1)):
            raise RuntimeError(f"accepted-row count mismatch for pool {pool}")
        for row in accepted_rows:
            path = row.get("image_path", "")
            if not path:
                raise RuntimeError(f"empty accepted image path in pool {pool}")
            if path in rejected_paths:
                conflicts["additional_row_for_rejected_path"] += 1
                continue
            label = row["body_type" if args.head == "body" else "color"]
            previous = combined.get(path)
            if previous is not None:
                previous_label = previous["body_type" if args.head == "body" else "color"]
                if previous_label != label:
                    conflicts["conflicting_duplicate_label"] += 1
                    del combined[path]
                    pool_counts[previous["combined_teacher_source_pool"]] -= 1
                    label_counts[previous_label] -= 1
                    rejected_paths.add(path)
                else:
                    conflicts["identical_duplicate_path"] += 1
                continue
            output = dict(row)
            output["combined_teacher_source_pool"] = pool
            combined[path] = output
            pool_counts[pool] += 1
            label_counts[label] += 1
        source_evidence.append({
            "pool": pool,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "report_status": report.get("status"),
            "accepted_rows": len(accepted_rows),
        })

    rows = [combined[path] for path in sorted(combined)]
    if not rows:
        raise RuntimeError("no accepted rows survived teacher-manifest combination")
    if "combined_teacher_source_pool" not in fields:
        fields.append("combined_teacher_source_pool")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if args.head == "body":
        policy = {
            "all_teachers_exact_agreement": True,
            "all_views_exact_agreement": True,
            "official_coarse_class_compatibility_required": True,
            "frozen_video_used": False,
        }
    else:
        policy = {
            "foreground_or_official_labels_retained_only_not_replaced": True,
            "rejected_rows_changed_to_unknown_unsupervised": True,
            "model_predictions_used_to_create_new_labels": False,
            "test_split_not_used": True,
            "frozen_video_not_used": True,
        }
    policy.update({
        "all_inputs_hash_verified": True,
        "only_previously_accepted_rows_retained": True,
        "conflicting_duplicate_paths_rejected": True,
        "train_only": True,
        "validation_or_test_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    })
    report = {
        "schema_version": f"combined-{args.head}-teacher-consensus-manifests-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "head": args.head,
        "sources": source_evidence,
        "accepted_rows": len(rows),
        "accepted_pool_counts": dict(sorted(pool_counts.items())),
        "accepted_label_counts": dict(sorted(label_counts.items())),
        "conflict_counts": dict(sorted(conflicts.items())),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": policy,
        "decision": "eligible only for downstream combined curation; no teacher gate was relaxed",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "head": args.head,
        "accepted_rows": len(rows),
        "pools": dict(sorted(pool_counts.items())),
        "labels": dict(sorted(label_counts.items())),
        "conflicts": dict(sorted(conflicts.items())),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
