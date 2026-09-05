#!/usr/bin/env python3
"""Reconcile full train-only scene audit with prior agent visual decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y"}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48", "frozen_video")


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_key(value: str) -> str:
    return os.path.normcase(os.path.normpath(str(value or "").strip()))


def classification(
    row: dict[str, str],
    stage223: dict[str, dict[str, str]],
    stage239: dict[str, dict[str, str]],
) -> tuple[str, str, bool]:
    """Return census tier, reason, and quota eligibility."""
    source = str(row.get("source_dataset") or "").strip()
    key = path_key(row.get("image_path", ""))
    if source == "NightOwls":
        if (
            truthy(row.get("research_only"))
            and not truthy(row.get("deployment_eligible"))
            and truthy(row.get("color_supervised"))
            and str(row.get("scene_label") or "").startswith("source_confirmed_night")
        ):
            return "verified_real_night", "NightOwls source truth plus Stage241 agent-visible color acceptance", True
        return "unknown", "NightOwls row missing complete Stage241 acceptance contract", False

    visual223 = stage223.get(key)
    if visual223:
        decision = str(visual223.get("visual_decision") or "")
        scene = str(visual223.get("visual_scene") or "")
        eligible = truthy(visual223.get("eligible_for_real_night_color_quota"))
        if decision == "accept" and scene == "night" and eligible:
            return "verified_real_night", "Stage223 agent visual acceptance", True
        if decision == "accept" and scene == "low_light":
            return "verified_real_low_light", "Stage223 agent visual acceptance", True
        return "reviewed_not_usable_as_night", f"Stage223:{visual223.get('visual_reason', 'rejected')}", False

    visual239 = stage239.get(key)
    if visual239:
        if truthy(visual239.get("training_eligible_from_stage239")):
            return "verified_real_night", "Stage239 agent visual acceptance", True
        return "reviewed_not_usable_as_night", f"Stage239:{visual239.get('review_reason', 'rejected')}", False

    stage_scene = str(row.get("stage177_scene_label") or "unknown")
    origin = str(row.get("stage177_scene_origin") or "")
    machine_night = truthy(row.get("stage177_machine_night_candidate"))
    if stage_scene == "daylight":
        return "daylight", origin or "scene audit daylight", False
    if stage_scene == "low_light" or machine_night:
        return "probable_low_light_unverified", origin or "machine low-light candidate", False
    if stage_scene == "night":
        return "metadata_night_unverified", origin or "night metadata not visually reviewed", False
    return "unknown", origin or "missing scene evidence", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--stage248-report", type=Path, required=True)
    parser.add_argument("--expected-stage248-report-sha256", required=True)
    parser.add_argument("--stage223-overlay", type=Path, required=True)
    parser.add_argument("--expected-stage223-overlay-sha256", required=True)
    parser.add_argument("--stage239-overlay", type=Path, required=True)
    parser.add_argument("--expected-stage239-overlay-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite Stage249 evidence: {args.output_root}")
    require_sha(args.manifest, args.expected_manifest_sha256, "Stage248 enriched manifest")
    require_sha(args.stage248_report, args.expected_stage248_report_sha256, "Stage248 report")
    require_sha(args.stage223_overlay, args.expected_stage223_overlay_sha256, "Stage223 overlay")
    require_sha(args.stage239_overlay, args.expected_stage239_overlay_sha256, "Stage239 overlay")
    report248 = json.loads(args.stage248_report.read_text(encoding="utf-8"))
    policy248 = report248["policy"]
    if policy248.get("train_pixels_only") is not True or policy248.get("frozen_video_used") is not False:
        raise RuntimeError("Stage248 isolation policy mismatch")
    if report248.get("scope", {}).get("validation_or_test_images_opened") != 0:
        raise RuntimeError("Stage248 opened validation/test pixels")

    stage223_rows = read_csv(args.stage223_overlay)
    stage239_rows = read_csv(args.stage239_overlay)
    stage223 = {path_key(row.get("image_path", "")): row for row in stage223_rows}
    stage239 = {path_key(row.get("image_path", "")): row for row in stage239_rows}

    rows = read_csv(args.manifest)
    output_rows: list[dict[str, str]] = []
    tiers: Counter[str] = Counter()
    tier_sources: Counter[tuple[str, str]] = Counter()
    tier_colors: Counter[tuple[str, str]] = Counter()
    quota_rows = 0
    effective_rows = 0
    read_errors = 0
    seen_paths: set[str] = set()
    groups_by_tier: dict[str, set[str]] = defaultdict(set)

    for row_number, row in enumerate(rows, 2):
        searchable = " ".join(str(row.get(field) or "") for field in ("image_path", "video_id", "source_frame_id")).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"forbidden frozen marker at manifest row {row_number}")
        if str(row.get("split") or "").strip().lower() != "train":
            continue
        if not truthy(row.get("stage177_effective_representative")):
            continue
        if not truthy(row.get("color_supervised")):
            continue
        effective_rows += 1
        if row.get("stage177_read_error"):
            read_errors += 1
        image_key = path_key(row.get("image_path", ""))
        if image_key in seen_paths:
            raise RuntimeError(f"duplicate effective image path: {image_key}")
        seen_paths.add(image_key)
        tier, reason, eligible = classification(row, stage223, stage239)
        source = str(row.get("source_dataset") or "unknown")
        color = str(row.get("color") or "unknown")
        group = str(row.get("stage177_group_key") or row.get("track_key") or row.get("video_id") or image_key)
        tiers[tier] += 1
        tier_sources[(tier, source)] += 1
        tier_colors[(tier, color)] += 1
        groups_by_tier[tier].add(group)
        quota_rows += int(eligible)
        output_rows.append({
            "stage249_manifest_row": str(row_number),
            "image_path": row.get("image_path", ""),
            "source_dataset": source,
            "camera_id": row.get("camera_id", ""),
            "video_id": row.get("video_id", ""),
            "track_group": row.get("track_group", ""),
            "stage177_group_key": row.get("stage177_group_key", ""),
            "color": color,
            "stage248_scene_label": row.get("stage177_scene_label", ""),
            "stage248_scene_origin": row.get("stage177_scene_origin", ""),
            "stage249_scene_tier": tier,
            "stage249_reason": reason,
            "stage249_verified_adverse_color_quota_eligible": str(eligible).lower(),
        })

    args.output_root.mkdir(parents=True, exist_ok=False)
    overlay_path = args.output_root / "stage249-full-color-scene-census.csv"
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]) if output_rows else ["image_path"])
        writer.writeheader()
        writer.writerows(output_rows)

    verified_night = tiers["verified_real_night"]
    verified_low = tiers["verified_real_low_light"]
    verified_adverse = verified_night + verified_low
    report = {
        "schema_version": "stage249-full-color-scene-census-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_conservative_full_train_census",
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "stage248_report": str(args.stage248_report.resolve()),
            "stage248_report_sha256": sha256(args.stage248_report),
            "stage223_overlay": str(args.stage223_overlay.resolve()),
            "stage223_overlay_sha256": sha256(args.stage223_overlay),
            "stage239_overlay": str(args.stage239_overlay.resolve()),
            "stage239_overlay_sha256": sha256(args.stage239_overlay),
        },
        "scope": {
            "effective_color_supervised_train_rows": effective_rows,
            "unique_effective_image_paths": len(seen_paths),
            "read_errors": read_errors,
            "validation_or_test_rows_used": 0,
            "image_pixels_opened_by_stage249": 0,
        },
        "census": {
            "tier_counts": dict(sorted(tiers.items())),
            "tier_group_counts": {tier: len(groups) for tier, groups in sorted(groups_by_tier.items())},
            "verified_real_night_rows": verified_night,
            "verified_real_low_light_rows": verified_low,
            "verified_adverse_light_rows": verified_adverse,
            "verified_night_fraction": verified_night / effective_rows if effective_rows else 0.0,
            "verified_adverse_light_fraction": verified_adverse / effective_rows if effective_rows else 0.0,
            "required_adverse_light_fraction": 0.30,
            "adverse_light_quota_met": verified_adverse / effective_rows >= 0.30 if effective_rows else False,
            "quota_eligible_rows_crosscheck": quota_rows,
        },
        "tier_source_counts": {f"{tier}|{source}": count for (tier, source), count in sorted(tier_sources.items())},
        "tier_color_counts": {f"{tier}|{color}": count for (tier, color), count in sorted(tier_colors.items())},
        "interpretation": {
            "verified_is_lower_bound_not_total_real_night": True,
            "unknown_is_not_daylight": True,
            "metadata_night_is_not_visual_truth": True,
            "machine_low_light_is_not_confirmed_night": True,
            "row_counts_are_not_independent_vehicle_counts": True,
            "next_action": "semantically triage all unresolved train rows, visually audit every high-confidence real-night group, and acquire new recording-diverse night supervision",
        },
        "outputs": {"overlay": str(overlay_path), "overlay_sha256": sha256(overlay_path)},
        "policy": {
            "train_only": True,
            "prior_agent_decisions_preserved": True,
            "uncertain_rows_remain_unknown_or_probable": True,
            "no_scene_or_color_truth_force_assigned": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage249-full-color-scene-census.json"
    atomic_json(report_path, report)
    for path in (overlay_path, report_path):
        Path(str(path) + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scope": report["scope"], "census": report["census"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
