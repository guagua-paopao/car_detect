#!/usr/bin/env python3
"""Materialize the fail-closed Stage239 visual review of Stage238 candidates.

The Stage238 CLIP score is used only to rank train-split images for review.  This
script records the agent's contact-sheet review plus original-resolution checks
for the seven ambiguous Open Images candidates.  It never changes the source
manifest or promotes an unreviewed semantic prediction to truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ACCEPTED = {
    ("Open-Images-V7", 1): {
        "reviewed_scene": "night",
        "review_decision": "accept_confirmed_real_night_visible_color",
        "review_reason": "outdoor night scene; artificial lighting and dark sky/background; red bus body remains visible",
        "original_resolution_reviewed": True,
    },
    ("Open-Images-V7", 2): {
        "reviewed_scene": "night",
        "review_decision": "accept_confirmed_real_night_visible_color",
        "review_reason": "wet roadway under artificial illumination at night; red vehicle body remains visible",
        "original_resolution_reviewed": True,
    },
}

ORIGINAL_REVIEWED_REJECTIONS = {
    ("Open-Images-V7", 3): "extremely small crop; scene and color are not reliably judgeable",
    ("Open-Images-V7", 5): "daylight archival/cast image, not a real night scene",
    ("Open-Images-V7", 44): "snow/rain daylight with headlights, not confirmed night",
    ("Open-Images-V7", 51): "very small motion-blurred crop; color and night scene are not reliable",
    ("Open-Images-V7", 63): "rainy daylight/overcast crop with strong blur, not confirmed night",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_rejection(source: str) -> tuple[str, str]:
    if source == "DVM-CAR-2.0":
        return "catalog_or_non_scene", "catalog cutout, part, screen, or advertisement; not a real night camera scene"
    if source == "Vehicle-Rear":
        return "nonconfirmed_scene", "flash/indoor/bright vehicle-rear image; no reliable outdoor night context"
    if source == "BMD-45-RAW":
        return "daylight_or_nonconfirmed", "reviewed CCTV crop is daylight or lacks reliable evidence of a night scene"
    return "daylight_or_ambiguous", "contact-sheet review did not confirm a real night scene with reliable visible color"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--previous-active-night", type=int, default=25)
    parser.add_argument("--previous-active-lowlight", type=int, default=3)
    parser.add_argument("--active-pool-rows", type=int, default=118975)
    parser.add_argument("--isolated-exdark-night-rows", type=int, default=3)
    parser.add_argument("--isolated-nightowls-night-rows", type=int, default=27)
    parser.add_argument("--isolated-nightowls-tracks", type=int, default=9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.ranking.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 218:
        raise RuntimeError(f"expected 218 ranked rows, got {len(rows)}")

    overlay = []
    accepted = []
    reviewed_sources = Counter()
    rejected_reasons = Counter()
    for row in rows:
        source = row["source_dataset"]
        rank = int(row["source_rank"])
        key = (source, rank)
        reviewed_sources[source] += 1
        out = dict(row)
        if key in ACCEPTED:
            decision = ACCEPTED[key]
            out.update(decision)
            out["agent_reviewed"] = "true"
            out["training_eligible_from_stage239"] = "true"
            accepted.append(out)
        else:
            if key in ORIGINAL_REVIEWED_REJECTIONS:
                scene, reason = "unknown", ORIGINAL_REVIEWED_REJECTIONS[key]
                original = True
            else:
                scene, reason = default_rejection(source)
                original = False
            out.update(
                {
                    "reviewed_scene": scene,
                    "review_decision": "reject_fail_closed",
                    "review_reason": reason,
                    "original_resolution_reviewed": str(original).lower(),
                    "agent_reviewed": "true",
                    "training_eligible_from_stage239": "false",
                }
            )
            rejected_reasons[scene] += 1
        overlay.append(out)

    fieldnames = list(overlay[0].keys())
    overlay_path = args.output_dir / "stage239-unknown-night-agent-review-overlay.csv"
    with overlay_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overlay)

    accepted_path = args.output_dir / "stage239-confirmed-night-visible-color.csv"
    with accepted_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)

    decisions_path = args.output_dir / "stage239-agent-decisions.json"
    decisions = {
        "review_scope": "all 218 Stage238 contact-sheet candidates; seven ambiguous Open Images crops also opened at original resolution",
        "accepted_keys": [f"{source}|rank={rank}" for source, rank in ACCEPTED],
        "original_resolution_rejected_keys": [f"{source}|rank={rank}" for source, rank in ORIGINAL_REVIEWED_REJECTIONS],
        "default_rejection_policy": {
            "DVM-CAR-2.0": "reject catalog/parts/screens/advertisements as non-real-camera scenes",
            "Vehicle-Rear": "reject flash/indoor/bright rear images without confirmed outdoor night context",
            "BMD-45-RAW": "reject reviewed daylight/nonconfirmed CCTV crops",
            "Open-Images-V7": "reject unless the visual review confirms real night and visible vehicle color",
        },
        "semantic_rank_is_not_truth": True,
    }
    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    active_night = args.previous_active_night + len(accepted)
    active_lowlight = args.previous_active_lowlight
    active_adverse = active_night + active_lowlight
    all_confirmed_night_rows = active_night + args.isolated_exdark_night_rows + args.isolated_nightowls_night_rows
    report = {
        "schema_version": "stage239-unknown-night-agent-review-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_fail_closed_conservative_recount",
        "ranked_candidates_reviewed": len(rows),
        "reviewed_source_counts": dict(sorted(reviewed_sources.items())),
        "original_resolution_candidates_reviewed": 7,
        "accepted_confirmed_real_night_visible_color_rows": len(accepted),
        "accepted_source_counts": dict(Counter(r["source_dataset"] for r in accepted)),
        "accepted_color_counts": dict(Counter(r["color"] for r in accepted)),
        "rejected_rows": len(rows) - len(accepted),
        "rejected_scene_reason_counts": dict(sorted(rejected_reasons.items())),
        "active_track_balanced_color_pool": {
            "rows": args.active_pool_rows,
            "previous_confirmed_night_rows": args.previous_active_night,
            "new_confirmed_night_rows": len(accepted),
            "confirmed_night_rows": active_night,
            "confirmed_lowlight_rows": active_lowlight,
            "confirmed_adverse_light_rows": active_adverse,
            "confirmed_night_fraction": active_night / args.active_pool_rows,
            "confirmed_adverse_light_fraction": active_adverse / args.active_pool_rows,
            "interpretation": "exact conservative lower bound after full pixel audit and ranked visual review; not proof that every unknown row is daylight",
        },
        "isolated_research_only_confirmed_night_rows": {
            "ExDark": args.isolated_exdark_night_rows,
            "NightOwls_rows": args.isolated_nightowls_night_rows,
            "NightOwls_tracks": args.isolated_nightowls_tracks,
            "not_merged_into_active_pool": True,
        },
        "all_audited_pools_confirmed_night_rows": all_confirmed_night_rows,
        "all_audited_pools_confirmed_adverse_light_rows": all_confirmed_night_rows + active_lowlight,
        "counting_warning": "row counts are not independent vehicle counts; NightOwls has 27 rows from only 9 tracks, and research-only sources remain isolated",
        "outputs": {
            "overlay": str(overlay_path),
            "overlay_sha256": sha256(overlay_path),
            "accepted_manifest": str(accepted_path),
            "accepted_manifest_sha256": sha256(accepted_path),
            "decisions": str(decisions_path),
            "decisions_sha256": sha256(decisions_path),
        },
        "policy": {
            "all_ranked_contact_sheets_reviewed": True,
            "semantic_ranking_promoted_directly_to_truth": False,
            "uncertain_rows_rejected": True,
            "source_manifest_modified": False,
            "training_started": False,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage239-unknown-night-agent-review.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sum_paths = [args.ranking, decisions_path, overlay_path, accepted_path, report_path]
    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sum_paths), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
