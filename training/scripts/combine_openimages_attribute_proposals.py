#!/usr/bin/env python3
"""Combine hash-verified train-only unlabeled Open Images attribute proposal pools."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.manifest) == len(args.report) == len(args.pool)):
        raise ValueError("--manifest, --report and --pool counts must match")
    if len(set(args.pool)) != len(args.pool):
        raise ValueError("proposal pool names must be unique")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite combined proposal evidence")

    allowed_statuses = {
        "pass_geometry_pending_multiteacher_review",
        "pass_photometric_pending_multiteacher_review",
    }
    fields: list[str] = []
    rows = []
    paths = set()
    source_evidence = []
    pool_counts = Counter()
    for manifest_path, report_path, pool in zip(args.manifest, args.report, args.pool):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in allowed_statuses:
            raise RuntimeError(f"proposal report did not pass for pool {pool}")
        if report.get("output_manifest_sha256") != sha256(manifest_path):
            raise RuntimeError(f"proposal manifest hash mismatch for pool {pool}")
        policy = report.get("policy", {})
        if not (
            policy.get("attributes_unsupervised_until_multiteacher_consensus") is True
            and policy.get("frozen_video_used") is False
            and policy.get("production_model_modified") is False
            and policy.get("deployment_performed") is False
        ):
            raise RuntimeError(f"proposal policy is not isolated for pool {pool}")
        manifest_fields, manifest_rows = load(manifest_path)
        for field in manifest_fields:
            if field not in fields:
                fields.append(field)
        for row in manifest_rows:
            if row.get("split") != "train":
                raise RuntimeError(f"non-train row in proposal pool {pool}")
            if row.get("body_type") not in {"", "unknown"} or row.get("color") not in {"", "unknown"}:
                raise RuntimeError(f"preassigned attribute label in proposal pool {pool}")
            if truthy(row.get("body_type_supervised")) or truthy(row.get("color_supervised")):
                raise RuntimeError(f"supervised row entered unlabeled proposal pool {pool}")
            path = row.get("image_path", "")
            if not path or path in paths:
                raise RuntimeError(f"empty or duplicate proposal image path in pool {pool}")
            paths.add(path)
            output = dict(row)
            output["proposal_source_pool"] = pool
            rows.append(output)
            pool_counts[pool] += 1
        source_evidence.append({
            "pool": pool,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "rows": len(manifest_rows),
        })

    rows.sort(key=lambda row: (row["proposal_source_pool"], row["image_path"]))
    if "proposal_source_pool" not in fields:
        fields.append("proposal_source_pool")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    night = sum(truthy(row.get("night")) for row in rows)
    small = sum(row.get("vehicle_size") == "small" for row in rows)
    occluded = sum(truthy(row.get("occluded")) for row in rows)
    truncated = sum(truthy(row.get("truncated")) for row in rows)
    report = {
        "schema_version": "openimages-combined-unlabeled-attribute-proposals-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "sources": source_evidence,
        "rows": len(rows),
        "unique_image_paths": len(paths),
        "pool_counts": dict(sorted(pool_counts.items())),
        "night_rows": night,
        "night_fraction": night / len(rows) if rows else 0.0,
        "small_rows": small,
        "occluded_rows": occluded,
        "truncated_rows": truncated,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "all_inputs_hash_verified": True,
            "all_rows_train_only": True,
            "all_attributes_remain_unknown_unsupervised": True,
            "duplicate_image_paths_rejected": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "run strict body teachers and foreground-color proposal plus color-teacher audit",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "rows": len(rows),
        "pools": dict(sorted(pool_counts.items())),
        "night": night,
        "small": small,
        "occluded": occluded,
        "truncated": truncated,
        "output_sha256": report["output_manifest_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
