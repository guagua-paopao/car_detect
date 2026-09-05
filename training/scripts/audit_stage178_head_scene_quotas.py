#!/usr/bin/env python3
"""Summarize Stage177 night/low-light ratios for each supervised head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}
COUNTED_ADVERSE_ORIGINS = {
    "existing_explicit_night",
    "existing_explicit_low_light",
    "machine_track_night_candidate",
    "machine_track_lowlight_consensus",
}


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def supervised(row: dict[str, str], head: str) -> bool:
    value = str(row.get(f"{head}_supervised") or "").strip()
    return True if not value else truthy(value)


def selected_scopes(row: dict[str, str], bodies: set[str], colors: set[str]) -> list[str]:
    body = str(row.get("body_type") or "").strip().lower()
    color = str(row.get("color") or "").strip().lower()
    coarse = str(row.get("coarse_body_family") or "").strip().lower()
    exact_body = supervised(row, "body_type") and body in bodies
    exact_color = supervised(row, "color") and color in colors
    scopes = ["loader_union"]
    if exact_body:
        scopes.append("body_exact")
    if exact_body or coarse in {"car", "truck"}:
        scopes.append("body_exact_or_coarse")
    if exact_color:
        scopes.append("color_exact")
    return scopes


def finish(counter: Counter[str], target: float) -> dict[str, object]:
    total = counter["total"]
    counted = counter["effective_adverse"]
    positive_additions = max(0, math.ceil((target * total - counted) / (1.0 - target))) if total else 0
    return {
        "total": total,
        "existing_explicit_night": counter["explicit_night"],
        "existing_explicit_low_light": counter["explicit_low_light"],
        "machine_track_low_light_proxy": counter["machine_track"],
        "machine_single_frame_candidate_excluded_from_quota": counter["machine_single"],
        "unknown": counter["unknown"],
        "effective_adverse_light": counted,
        "effective_adverse_light_fraction": counted / total if total else 0.0,
        "explicit_night_fraction": counter["explicit_night"] / total if total else 0.0,
        "unknown_fraction": counter["unknown"] / total if total else 0.0,
        "target_fraction": target,
        "passes_target": counted / total >= target if total else False,
        "minimum_all_positive_additions_to_reach_final_target": positive_additions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_sha = sha256_file(args.manifest)
    labels_sha = sha256_file(args.labels)
    if manifest_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError("manifest SHA256 mismatch")
    if labels_sha.lower() != args.expected_labels_sha256.lower():
        raise RuntimeError("labels SHA256 mismatch")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    bodies = {str(value).strip().lower() for value in labels["body_types"]} - {"unknown"}
    colors = {str(value).strip().lower() for value in labels["colors"]} - {"unknown"}
    states = {
        "raw_loader_rows": lambda row: bool(row.get("stage177_scene_label")),
        "dedup_eligible_images": lambda row: truthy(row.get("stage177_dedup_eligible")),
        "track_balanced_effective_images": lambda row: truthy(row.get("stage177_effective_representative")),
    }
    counters: dict[str, dict[str, Counter[str]]] = {
        state: {scope: Counter() for scope in ("loader_union", "body_exact", "body_exact_or_coarse", "color_exact")}
        for state in states
    }
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split") or "").strip().lower() != "train":
                continue
            origin = str(row.get("stage177_scene_origin") or "")
            scene = str(row.get("stage177_scene_label") or "")
            for state, predicate in states.items():
                if not predicate(row):
                    continue
                for scope in selected_scopes(row, bodies, colors):
                    counter = counters[state][scope]
                    counter["total"] += 1
                    counter["explicit_night"] += origin == "existing_explicit_night"
                    counter["explicit_low_light"] += origin == "existing_explicit_low_light"
                    counter["machine_track"] += origin.startswith("machine_track_")
                    counter["machine_single"] += origin.startswith("machine_single_frame_")
                    counter["unknown"] += scene == "unknown"
                    counter["effective_adverse"] += origin in COUNTED_ADVERSE_ORIGINS
    report = {
        "schema_version": "stage178-head-scene-quota-audit-v1",
        "status": "pass_audit_quota_unmet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"manifest": str(args.manifest), "sha256": manifest_sha, "labels_sha256": labels_sha},
        "target_fraction": args.target,
        "states": {
            state: {scope: finish(counter, args.target) for scope, counter in scopes.items()}
            for state, scopes in counters.items()
        },
        "policy": {
            "metadata_only": True,
            "images_opened": False,
            "single_frame_darkness_excluded_from_quota": True,
            "machine_track_consensus_may_count_only_as_low_light_proxy": True,
            "source_confirmed_night_not_inferred_from_pixels": True,
            "training_started": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "effective": report["states"]["track_balanced_effective_images"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
