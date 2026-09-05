#!/usr/bin/env python3
"""Materialize Stage241's fail-closed agent review of NightOwls tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-tracks-per-recording-color", type=int, default=2)
    parser.add_argument("--minimum-training-tracks", type=int, default=80)
    parser.add_argument("--minimum-training-colors", type=int, default=5)
    parser.add_argument("--minimum-tracks-per-color", type=int, default=8)
    args = parser.parse_args()

    if args.output_dir.exists():
        existing = {path.resolve() for path in args.output_dir.iterdir()}
        allowed = {args.decisions.resolve()}
        if existing != allowed:
            raise FileExistsError(f"refusing to overwrite evidence: {args.output_dir}")
    candidate_sha = sha256(args.candidates)
    if candidate_sha.lower() != args.expected_candidates_sha256.lower():
        raise RuntimeError(f"candidate manifest SHA mismatch: {candidate_sha}")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    accepted_by_color = decisions["accepted_audit_ids_by_color"]
    accepted_map: dict[str, str] = {}
    for color, audit_ids in accepted_by_color.items():
        for audit_id in audit_ids:
            if audit_id in accepted_map:
                raise RuntimeError(f"duplicate accepted audit id: {audit_id}")
            accepted_map[audit_id] = color

    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["stage240_audit_id"]].append(row)
    if len(rows) != 912 or len(grouped) != 304:
        raise RuntimeError(f"unexpected Stage240 shape: rows={len(rows)} tracks={len(grouped)}")
    if any(len(track_rows) != 3 for track_rows in grouped.values()):
        raise RuntimeError("every Stage240 track must have exactly three representatives")
    missing = sorted(set(accepted_map) - set(grouped))
    if missing:
        raise RuntimeError(f"accepted audit ids missing from candidates: {missing}")

    accepted_tracks = []
    for audit_id, color in accepted_map.items():
        track_rows = grouped[audit_id]
        first = track_rows[0]
        accepted_tracks.append(
            {
                "audit_id": audit_id,
                "color": color,
                "recording_id": first["recording_id"],
                "quality": float(first["stage240_average_quality"]),
                "rows": track_rows,
            }
        )

    by_recording_color: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for track in accepted_tracks:
        by_recording_color[(str(track["recording_id"]), str(track["color"]))].append(track)
    eligible_ids: set[str] = set()
    capped_ids: set[str] = set()
    for tracks in by_recording_color.values():
        tracks.sort(key=lambda track: (-float(track["quality"]), str(track["audit_id"])))
        for index, track in enumerate(tracks):
            audit_id = str(track["audit_id"])
            if index < args.maximum_tracks_per_recording_color:
                eligible_ids.add(audit_id)
            else:
                capped_ids.add(audit_id)

    output_rows = []
    training_rows = []
    for row in rows:
        out = dict(row)
        audit_id = row["stage240_audit_id"]
        color = accepted_map.get(audit_id, "unknown")
        visually_accepted = audit_id in accepted_map
        training_eligible = audit_id in eligible_ids
        out.update(
            {
                "stage241_agent_color": color,
                "stage241_agent_decision": "accepted_visible_stable_color" if visually_accepted else "rejected_to_unknown_fail_closed",
                "stage241_recording_group": f"NightOwls|train|recording_{int(row['recording_id']):02d}",
                "stage241_recording_color_cap_passed": str(training_eligible).lower(),
                "color": color if visually_accepted else "unknown",
                "color_supervised": str(training_eligible).lower(),
                "formal_train_eligible": str(training_eligible).lower(),
                "stage241_training_role": "supervised_research_only_night_color_repair" if training_eligible else "unlabeled_night_consistency_only",
            }
        )
        output_rows.append(out)
        if training_eligible:
            training_rows.append(out)

    fieldnames = list(output_rows[0].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "stage241-nightowls-color-agent-overlay.csv"
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    training_path = args.output_dir / "stage241-nightowls-color-supervised-research-only.csv"
    with training_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_rows)

    visual_counts = Counter(accepted_map.values())
    training_track_counts = Counter(accepted_map[audit_id] for audit_id in eligible_ids)
    training_row_counts = Counter(row["stage241_agent_color"] for row in training_rows)
    quota_pass = (
        len(eligible_ids) >= args.minimum_training_tracks
        and len(training_track_counts) >= args.minimum_training_colors
        and min(training_track_counts.values(), default=0) >= args.minimum_tracks_per_color
    )
    report = {
        "schema_version": "stage241-nightowls-color-agent-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_targeted_research_candidate_authorized" if quota_pass else "complete_fail_closed_insufficient_quota",
        "inputs": {
            "candidates": str(args.candidates),
            "candidates_sha256": candidate_sha,
            "decisions": str(args.decisions),
            "decisions_sha256": sha256(args.decisions),
        },
        "reviewed_tracks": len(grouped),
        "reviewed_rows": len(rows),
        "visually_accepted_tracks": len(accepted_map),
        "visually_accepted_track_color_counts": dict(sorted(visual_counts.items())),
        "rejected_to_unknown_tracks": len(grouped) - len(accepted_map),
        "recording_color_cap": args.maximum_tracks_per_recording_color,
        "capped_duplicate_or_overrepresented_tracks": len(capped_ids),
        "training_tracks": len(eligible_ids),
        "training_track_color_counts": dict(sorted(training_track_counts.items())),
        "training_rows": len(training_rows),
        "training_row_color_counts": dict(sorted(training_row_counts.items())),
        "recording_groups": len({row["stage241_recording_group"] for row in training_rows}),
        "quota": {
            "minimum_training_tracks": args.minimum_training_tracks,
            "minimum_training_colors": args.minimum_training_colors,
            "minimum_tracks_per_color": args.minimum_tracks_per_color,
            "pass": quota_pass,
        },
        "outputs": {
            "overlay": str(overlay_path),
            "overlay_sha256": sha256(overlay_path),
            "supervised_manifest": str(training_path),
            "supervised_manifest_sha256": sha256(training_path),
        },
        "training_policy": {
            "authorized_use": "targeted low-weight night-color repair combined with the full existing supervised color manifest; never standalone training",
            "research_only": True,
            "deployment_eligible": False,
            "do_not_use_unaccepted_teacher_pseudolabels": True,
            "do_not_treat_rows_as_independent_vehicles": True,
            "split_or_sampling_group": "recording_id",
            "candidate_must_be_rejected_if_daylight_or_independent_night_validation_regresses": True,
        },
        "policy": {
            "train_split_only": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage241-nightowls-color-agent-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sum_paths = [args.candidates, args.decisions, overlay_path, training_path, report_path]
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sum_paths), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
