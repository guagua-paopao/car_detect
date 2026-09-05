#!/usr/bin/env python3
"""Materialize the agent visual decision over Stage235 color contact sheets."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage235-manifest", type=Path, required=True)
    parser.add_argument("--stage235-report", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage236 evidence")

    stage235 = json.loads(args.stage235_report.read_text(encoding="utf-8"))
    if stage235["output_manifest_sha256"] != sha256(args.stage235_manifest):
        raise RuntimeError("Stage235 manifest SHA256 mismatch")
    with args.stage235_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    # Every accepted candidate for these four colors appeared in its complete
    # contact sheet (all counts are below the Stage235 per-group limit of 40).
    accepted_classes = {"blue", "red", "silver", "white"}
    rejected_reasons = {
        "black": "class-level collapse and sampled sheet contains dark indeterminate or non-black vehicles",
        "yellow": "complete sheet is dominated by warm/sodium illumination and yellow-vs-cream is not reliable",
    }
    accepted_rows = Counter()
    accepted_tracks: dict[str, set[str]] = {}
    rejected_rows = Counter()
    audited_candidate_rows = 0
    for row in rows:
        if row.get("color_teacher_consensus") != "accepted" or not truthy(row.get("color_supervised")):
            row["agent_visual_status"] = "not_stage235_supervised_candidate"
            row["agent_visual_reason"] = "not_in_stage235_supervised_representatives"
            continue
        audited_candidate_rows += 1
        proposed = row.get("color", "unknown")
        if proposed in accepted_classes:
            row["agent_visual_status"] = "accepted"
            row["agent_visual_reason"] = "complete_color_contact_sheet_visually_consistent_and_body_color_visible"
            row["color_review_status"] = "stage236_agent_visual_complete_class_sheet_accepted"
            accepted_rows[proposed] += 1
            accepted_tracks.setdefault(proposed, set()).add(row.get("track_key", ""))
        else:
            row["agent_visual_status"] = "rejected"
            row["agent_visual_reason"] = rejected_reasons.get(
                proposed, "color_not_conservatively_approved_by_agent_visual_audit"
            )
            row["color"] = "unknown"
            row["color_supervised"] = "false"
            row["formal_train_eligible"] = "false"
            row["review_status"] = "rejected"
            row["color_review_status"] = "stage236_agent_visual_fail_closed"
            row["label_confidence"] = "unknown"
            row["color_teacher_consensus"] = "agent_rejected"
            rejected_rows[proposed] += 1

    output_fields = fields + [
        field for field in ("agent_visual_status", "agent_visual_reason") if field not in fields
    ]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    sheets = {}
    for path in sorted(args.contact_sheet_dir.glob("label_color_*.jpg")):
        sheets[path.name] = sha256(path)
    accepted_track_counts = {key: len(value) for key, value in sorted(accepted_tracks.items())}
    report = {
        "schema_version": "stage236-nightowls-color-agent-visual-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_fail_closed_insufficient_diverse_supervision",
        "stage235_supervised_candidates_reviewed": audited_candidate_rows,
        "complete_contact_sheet_colors_reviewed": ["blue", "red", "silver", "white", "yellow"],
        "sampled_contact_sheet_colors_reviewed": ["black"],
        "accepted_rows": sum(accepted_rows.values()),
        "accepted_color_counts": dict(sorted(accepted_rows.items())),
        "accepted_tracks": sum(accepted_track_counts.values()),
        "accepted_track_color_counts": accepted_track_counts,
        "rejected_rows": sum(rejected_rows.values()),
        "rejected_color_counts": dict(sorted(rejected_rows.items())),
        "class_decisions": {
            "blue": "accept_complete_sheet",
            "red": "accept_complete_sheet",
            "silver": "accept_complete_sheet",
            "white": "accept_complete_sheet",
            "black": "reject_all_fail_closed_due_domain_collapse_and_sampled_visual_errors",
            "yellow": "reject_all_fail_closed_due_warm_illumination_ambiguity",
        },
        "stage235_report": str(args.stage235_report.resolve()),
        "stage235_report_sha256": sha256(args.stage235_report),
        "stage235_manifest": str(args.stage235_manifest.resolve()),
        "stage235_manifest_sha256": sha256(args.stage235_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "contact_sheet_sha256": sheets,
        "interpretation": {
            "stage235_black_track_fraction": 117 / 128,
            "teacher_domain_collapse_detected": True,
            "nightowls_is_valid_real_night_inventory": True,
            "nightowls_does_not_provide_ground_truth_vehicle_color": True,
            "accepted_batch_is_not_large_or_diverse_enough_for_supervised_retraining": True,
            "training_decision": "do_not_train_from_stage236_increment; acquire licensed color-ground-truth surveillance data or use NightOwls only as unlabeled consistency data in a separately gated experiment",
        },
        "policy": {
            "research_only": True,
            "deployment_eligible": False,
            "uncertain_colors_demoted_to_unknown": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
