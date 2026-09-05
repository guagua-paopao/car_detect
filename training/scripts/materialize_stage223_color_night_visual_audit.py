#!/usr/bin/env python3
"""Materialize fail-closed Stage223 agent visual decisions into auditable CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"input must be a regular non-symlink file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_index(path: Path, expected_sha: str, expected_rows: int) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha.lower():
        raise RuntimeError(f"index SHA256 mismatch for {path}: {actual_sha}")
    payload = load_json(path)
    rows = payload.get("row_index") or []
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows in {path}, got {len(rows)}")
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(1, expected_rows + 1)):
        raise RuntimeError(f"non-contiguous or reordered index in {path}")
    policy = payload.get("policy") or {}
    if policy.get("test_accessed") is True or policy.get("frozen_video_used") is True:
        raise RuntimeError(f"forbidden test/frozen access recorded in {path}")
    return payload


def decide_rows(
    index_payload: dict[str, Any],
    decision: dict[str, Any],
    set_name: str,
) -> list[dict[str, Any]]:
    accepted = {int(value) for value in decision.get("accepted_indices") or []}
    accepted_night = {int(value) for value in decision.get("accepted_night_indices") or []}
    accepted_low_light = {int(value) for value in decision.get("accepted_low_light_indices") or []}
    if accepted_night & accepted_low_light:
        raise RuntimeError(f"scene decisions overlap for {set_name}")
    if accepted_night | accepted_low_light not in (set(), accepted):
        raise RuntimeError(f"accepted scene decisions do not cover accepted indices for {set_name}")
    overrides = {int(key): str(value) for key, value in (decision.get("rejection_overrides") or {}).items()}
    records: list[dict[str, Any]] = []
    for row in index_payload["row_index"]:
        index = int(row["index"])
        is_accepted = index in accepted
        if is_accepted:
            scene = "night" if index in accepted_night else "low_light" if index in accepted_low_light else "adverse_light"
            reason = "agent_visual_scene_and_color_confirmed"
        else:
            scene = "unknown"
            reason = overrides.get(index, str(decision["default_reason"]))
        records.append({
            "candidate_set": set_name,
            "index": index,
            "sha256": row.get("sha256"),
            "image_path": row.get("image_path"),
            "source_dataset": row.get("source_dataset", "VCoR" if set_name == "historical_novel" else "unknown"),
            "color": row.get("color"),
            "metadata_scene_claim": row.get("scene_claim", "night"),
            "visual_decision": "accept" if is_accepted else "reject",
            "visual_scene": scene,
            "visual_reason": reason,
            "eligible_for_real_night_color_quota": str(is_accepted and scene == "night").lower(),
            "eligible_for_adverse_light_color_training": str(is_accepted).lower(),
            "research_only": "true" if set_name == "historical_novel" else "false",
        })
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty audit CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--active-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    decisions = load_json(args.decisions.resolve())
    historical_spec = decisions["inputs"]["historical_novel_candidates"]
    active_spec = decisions["inputs"]["active_explicit_night_candidates"]
    historical = verify_index(
        args.historical_index.resolve(), historical_spec["index_sha256"], int(historical_spec["rows"])
    )
    active = verify_index(args.active_index.resolve(), active_spec["index_sha256"], int(active_spec["rows"]))

    if args.output_root.exists():
        existing = {path.name for path in args.output_root.iterdir()}
        allowed_existing = {args.decisions.resolve().name}
        if existing - allowed_existing:
            raise FileExistsError(f"refusing to overwrite non-decision outputs in {args.output_root}")
    else:
        args.output_root.mkdir(parents=True)

    rows = decide_rows(historical, historical_spec, "historical_novel")
    rows += decide_rows(active, active_spec, "active_explicit_night")
    overlay_path = args.output_root / "stage223-color-night-agent-visual-overlay.csv"
    write_csv(overlay_path, rows)

    accepted = [row for row in rows if row["visual_decision"] == "accept"]
    accepted_path = args.output_root / "stage223-color-night-accepted-index.csv"
    write_csv(accepted_path, accepted)
    rejected = [row for row in rows if row["visual_decision"] == "reject"]

    active_rows = [row for row in rows if row["candidate_set"] == "active_explicit_night"]
    novel_rows = [row for row in rows if row["candidate_set"] == "historical_novel"]
    accepted_night = [row for row in accepted if row["visual_scene"] == "night"]
    accepted_low_light = [row for row in accepted if row["visual_scene"] == "low_light"]
    report = {
        "schema_version": "stage223-color-night-agent-visual-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_fail_closed_real_scene_recount",
        "inputs": {
            "decisions": str(args.decisions.resolve()),
            "decisions_sha256": sha256_file(args.decisions.resolve()),
            "historical_index": str(args.historical_index.resolve()),
            "historical_index_sha256": historical_spec["index_sha256"],
            "active_index": str(args.active_index.resolve()),
            "active_index_sha256": active_spec["index_sha256"],
        },
        "recount": {
            "visual_candidates_reviewed": len(rows),
            "historical_novel_candidates_reviewed": len(novel_rows),
            "historical_novel_accepted": sum(row["visual_decision"] == "accept" for row in novel_rows),
            "active_metadata_explicit_night_candidates_reviewed": len(active_rows),
            "active_confirmed_real_night_visible_color": len(accepted_night),
            "active_confirmed_real_low_light_visible_color": len(accepted_low_light),
            "active_confirmed_adverse_light_visible_color": len(accepted),
            "active_false_or_unusable_night_metadata": sum(row["visual_decision"] == "reject" for row in active_rows),
            "active_metadata_night_false_or_unusable_rate": sum(row["visual_decision"] == "reject" for row in active_rows) / len(active_rows),
            "confirmed_night_fraction_of_track_balanced_color_pool": len(accepted_night) / 118975,
            "confirmed_adverse_light_fraction_of_track_balanced_color_pool": len(accepted) / 118975,
            "accepted_color_counts": dict(sorted(Counter(row["color"] for row in accepted).items())),
            "accepted_scene_counts": dict(sorted(Counter(row["visual_scene"] for row in accepted).items())),
            "rejection_reason_counts": dict(sorted(Counter(row["visual_reason"] for row in rejected).items())),
        },
        "interpretation": {
            "previous_144_was_metadata_confirmed_not_agent_visual_confirmed": True,
            "previous_144_overstated_real_visible_night_color": True,
            "new_historical_dark_image_candidates_do_not_add_real_night_color_quota": True,
            "current_real_night_color_supervision_is_severely_insufficient": True,
            "training_decision": "do_not_start_new_color_retrain_from_these_candidates_alone",
        },
        "outputs": {
            "overlay": str(overlay_path),
            "overlay_sha256": sha256_file(overlay_path),
            "accepted_index": str(accepted_path),
            "accepted_index_sha256": sha256_file(accepted_path),
        },
        "policy": decisions["constraints"],
    }
    report_path = args.output_root / "stage223-color-night-agent-visual-audit.json"
    atomic_json(report_path, report)
    sums = [overlay_path, accepted_path, report_path, args.decisions.resolve()]
    (args.output_root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sums), encoding="utf-8"
    )
    print(json.dumps(report["recount"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
