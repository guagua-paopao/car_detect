#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


NIGHT = {2, 3, 4, 9, 13}
LOW_LIGHT = {1}
UNKNOWN = {15, 143, 144, 145, 180, 181, 182, 184}
COLOR_RELIABLE_NIGHT = {2, 3, 13}
COLOR_RELIABLE_LOW_LIGHT = {1}
COLOR_UNRELIABLE = {4, 9, 180, 181, 182, 184}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_decision(index: int) -> tuple[str, str]:
    if index in NIGHT:
        return "night", "agent_visual_clear_night_cues"
    if index in LOW_LIGHT:
        return "low_light", "agent_visual_low_exposure_time_of_day_not_provable"
    if index in UNKNOWN:
        return "unknown", "agent_visual_scene_time_or_illumination_ambiguous"
    return "daylight", "agent_visual_daylight_or_dark_vehicle_false_positive"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--stage223-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)

    source = json.loads(args.index.read_text(encoding="utf-8-sig"))
    previous = json.loads(args.stage223_report.read_text(encoding="utf-8-sig"))
    rows = source["row_index"]
    if len(rows) != 184 or {int(row["index"]) for row in rows} != set(range(1, 185)):
        raise ValueError("Stage230 index must contain every index from 1 through 184 exactly once")
    if any(row.get("source_dataset") not in {"BMD-45-RAW", "Open-Images-V7", "Vehicle-Rear"} for row in rows):
        raise ValueError("unexpected source in Stage230 index")

    decisions = []
    for row in rows:
        index = int(row["index"])
        scene, reason = scene_decision(index)
        if index in COLOR_RELIABLE_NIGHT or index in COLOR_RELIABLE_LOW_LIGHT:
            color_usable = True
            color_reason = "visible_body_color_not_dominated_by_artificial_colored_light"
        elif index in COLOR_UNRELIABLE:
            color_usable = False
            color_reason = "colored_illumination_or_crop_does_not_reliably_show_labeled_body_color"
        else:
            color_usable = scene == "daylight"
            color_reason = "existing_source_color_retained_but_not_counted_as_adverse_light" if color_usable else "scene_unknown_not_counted"
        decisions.append({
            "index": index,
            "pixel_sha256": row["pixel_sha256"],
            "source_dataset": row["source_dataset"],
            "image_path": row["image_path"],
            "group_key": row["group_key"],
            "existing_color": row["color"],
            "machine_scene_proposal": row["scene_origin"],
            "agent_scene": scene,
            "agent_scene_reason": reason,
            "color_reliable_for_scene_quota": color_usable and scene in {"night", "low_light"},
            "color_reliability_reason": color_reason,
        })

    decision_path = args.output_root / "stage231-machine-lowlight-agent-decisions.json"
    decision_path.write_text(json.dumps({
        "schema_version": "stage231-machine-lowlight-agent-decisions-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "reviewed_rows": len(decisions),
        "decisions": decisions,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overlay_path = args.output_root / "stage231-machine-lowlight-agent-overlay.csv"
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    scene_counts = Counter(item["agent_scene"] for item in decisions)
    reliable_scene_counts = Counter(
        item["agent_scene"] for item in decisions if item["color_reliable_for_scene_quota"]
    )
    false_positive_count = scene_counts["daylight"]
    previous_recount = previous["recount"]
    pool = round(
        previous_recount["active_confirmed_real_night_visible_color"]
        / previous_recount["confirmed_night_fraction_of_track_balanced_color_pool"]
    )
    previous_night = previous_recount["active_confirmed_real_night_visible_color"]
    previous_low_light = previous_recount["active_confirmed_real_low_light_visible_color"]
    current_night = previous_night + reliable_scene_counts["night"]
    current_low_light = previous_low_light + reliable_scene_counts["low_light"]
    current_adverse = current_night + current_low_light

    report = {
        "schema_version": "stage231-color-machine-lowlight-agent-visual-audit-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_fail_closed_real_scene_recount",
        "stage230_review": {
            "machine_candidates_reviewed": len(decisions),
            "scene_counts": dict(sorted(scene_counts.items())),
            "reliable_color_scene_counts": dict(sorted(reliable_scene_counts.items())),
            "daylight_dark_vehicle_false_positives": false_positive_count,
            "machine_candidate_false_positive_rate": false_positive_count / len(decisions),
            "colored_light_or_unreliable_crop_count": len(COLOR_UNRELIABLE),
            "unknown_scene_count": scene_counts["unknown"],
        },
        "cumulative_active_train_recount": {
            "track_balanced_color_pool": pool,
            "previous_confirmed_real_night_visible_color": previous_night,
            "previous_confirmed_real_low_light_visible_color": previous_low_light,
            "new_confirmed_real_night_visible_color": reliable_scene_counts["night"],
            "new_confirmed_real_low_light_visible_color": reliable_scene_counts["low_light"],
            "confirmed_real_night_visible_color": current_night,
            "confirmed_real_low_light_visible_color": current_low_light,
            "confirmed_adverse_light_visible_color": current_adverse,
            "confirmed_night_fraction": current_night / pool,
            "confirmed_adverse_light_fraction": current_adverse / pool,
        },
        "interpretation": {
            "confirmed_count_is_conservative_lower_bound_not_global_total": True,
            "metadata_night_was_overinclusive": True,
            "single_frame_luma_screen_was_dominated_by_dark_vehicle_false_positives": True,
            "unknown_scene_inventory_still_not_exhaustively_human_visible": True,
            "training_decision": "do_not_retrain_from_this_small_increment; continue licensed real-night acquisition",
        },
        "inputs": {
            "stage230_index": str(args.index.resolve()),
            "stage230_index_sha256": sha256(args.index),
            "stage223_report": str(args.stage223_report.resolve()),
            "stage223_report_sha256": sha256(args.stage223_report),
        },
        "outputs": {
            "decisions": str(decision_path.resolve()),
            "overlay": str(overlay_path.resolve()),
        },
        "policy": {
            "train_pixels_only": True,
            "group_keys_preserved": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage231-color-machine-lowlight-agent-visual-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path = args.output_root / "SHA256SUMS"
    with sums_path.open("w", encoding="utf-8") as handle:
        for path in (decision_path, overlay_path, report_path):
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
