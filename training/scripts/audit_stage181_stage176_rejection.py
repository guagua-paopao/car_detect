#!/usr/bin/env python3
"""Independently verify Stage176 failed closed without fabricated labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=14857)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_sha = sha256_file(args.manifest)
    report_sha = sha256_file(args.report)
    if manifest_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError("manifest SHA256 mismatch")
    if report_sha.lower() != args.expected_report_sha256.lower():
        raise RuntimeError("report SHA256 mismatch")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    tracks: set[str] = set()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["rows"] += 1
            counts["train_rows"] += row.get("split") == "train"
            counts["body_unknown"] += row.get("body_type") == "unknown"
            counts["color_unknown"] += row.get("color") == "unknown"
            counts["body_unsupervised"] += not truthy(row.get("body_type_supervised"))
            counts["color_unsupervised"] += not truthy(row.get("color_supervised"))
            counts["accepted_consensus"] += row.get("teacher_consensus") == "accepted"
            counts["accepted_review"] += row.get("review_status") == "accepted_stage176_multiteacher_track_consensus"
            counts["known_track_label"] += row.get("track_teacher_consensus_class") not in {"", "unknown"}
            tracks.add(str(row.get("track_group") or ""))
    gates = {
        "row_count_matches": counts["rows"] == args.expected_rows == report.get("audited_rows"),
        "all_rows_train": counts["train_rows"] == counts["rows"],
        "all_body_labels_unknown": counts["body_unknown"] == counts["rows"],
        "all_color_labels_unknown": counts["color_unknown"] == counts["rows"],
        "all_body_unsupervised": counts["body_unsupervised"] == counts["rows"],
        "all_color_unsupervised": counts["color_unsupervised"] == counts["rows"],
        "zero_accepted_rows": counts["accepted_consensus"] == counts["accepted_review"] == 0 == report.get("accepted_rows"),
        "zero_known_track_labels": counts["known_track_label"] == 0,
        "training_not_authorized": report.get("pseudo_label_training_authorized") is False,
        "no_validation_test_or_frozen_use": (
            report.get("policy", {}).get("validation_or_test_used") is False
            and report.get("policy", {}).get("frozen_video_used") is False
        ),
    }
    result = {
        "schema_version": "stage181-stage176-independent-rejection-audit-v1",
        "status": "pass_rejection_sealed" if all(gates.values()) else "fail",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
        "source_report": str(args.report.resolve()), "source_report_sha256": report_sha,
        "counts": dict(sorted(counts.items())), "unique_track_groups": len(tracks),
        "gates": gates,
        "decision": "do not merge exact pseudo labels or train from Stage176; retain rows only as unknown research-only consistency candidates",
        "training_started": False, "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"].startswith("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
