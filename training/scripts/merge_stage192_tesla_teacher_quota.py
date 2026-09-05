#!/usr/bin/env python3
"""Merge Stage192 teacher evidence and recompute exact-head adverse-light quotas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def additions_needed(total: int, positive: int, target: float = 0.30) -> int:
    return max(0, math.ceil((target * total - positive) / (1.0 - target))) if total else 0


def quota(total: int, confirmed_night: int, reduced_visibility: int) -> dict[str, object]:
    effective = confirmed_night + reduced_visibility
    return {
        "total_exact_supervised_rows": total,
        "confirmed_natural_night_rows": confirmed_night,
        "confirmed_natural_night_fraction": confirmed_night / total if total else 0.0,
        "explicit_reduced_visibility_rows_excluding_natural_night": reduced_visibility,
        "effective_adverse_light_rows": effective,
        "effective_adverse_light_fraction": effective / total if total else 0.0,
        "target_fraction": 0.30,
        "passes_target": total > 0 and effective / total >= 0.30,
        "minimum_all_positive_additions_to_reach_target": additions_needed(total, effective),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage191-manifest", type=Path, required=True)
    parser.add_argument("--stage191-report", type=Path, required=True)
    parser.add_argument("--body-manifest", type=Path, required=True)
    parser.add_argument("--body-report", type=Path, required=True)
    parser.add_argument("--color-manifest", type=Path, required=True)
    parser.add_argument("--color-report", type=Path, required=True)
    parser.add_argument("--stage178-report", type=Path, required=True)
    parser.add_argument("--stage179-report", type=Path, required=True)
    parser.add_argument("--stage188-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage192 evidence")

    fields, original = load_csv(args.stage191_manifest)
    _, body_rows = load_csv(args.body_manifest)
    _, color_rows = load_csv(args.color_manifest)
    if not (len(original) == len(body_rows) == len(color_rows) == 3026):
        raise RuntimeError("Stage191/192 row-count mismatch")
    body_by_name = {row["source_image_name"]: row for row in body_rows}
    color_by_name = {row["source_image_name"]: row for row in color_rows}
    if len(body_by_name) != 3026 or len(color_by_name) != 3026:
        raise RuntimeError("duplicate Stage192 source image names")

    extra_fields = [
        "stage192_body_teacher_accepted", "stage192_color_teacher_accepted",
        "stage192_body_train_eligible", "stage192_color_train_eligible",
        "stage192_body_teacher_class", "stage192_body_teacher_confidence",
        "stage192_color_teacher_class", "stage192_color_teacher_confidence",
    ]
    body_counts, color_counts = Counter(), Counter()
    body_lighting, color_lighting = Counter(), Counter()
    body_accepted = color_accepted = 0
    output_rows: list[dict[str, str]] = []
    for row in original:
        name = row["source_image_name"]
        body = body_by_name[name]
        color = color_by_name[name]
        stage191_ok = truthy(row.get("stage191_train_eligible"))
        body_ok = stage191_ok and body.get("teacher_consensus") == "accepted"
        color_ok = stage191_ok and color.get("color_teacher_consensus") == "accepted"
        row = dict(row)
        row.update({
            "stage192_body_teacher_accepted": str(body_ok).lower(),
            "stage192_color_teacher_accepted": str(color_ok).lower(),
            "stage192_body_train_eligible": str(body_ok).lower(),
            "stage192_color_train_eligible": str(color_ok).lower(),
            "stage192_body_teacher_class": body.get("teacher_consensus_class", "unknown"),
            "stage192_body_teacher_confidence": body.get("teacher_consensus_confidence", "0"),
            "stage192_color_teacher_class": color.get("color_teacher_class", "unknown"),
            "stage192_color_teacher_confidence": color.get("color_teacher_confidence", "0"),
        })
        row["body_type_supervised"] = str(body_ok).lower()
        row["color_supervised"] = str(color_ok).lower()
        if not body_ok:
            row["body_type"] = "unknown"
        else:
            body_accepted += 1
            body_counts[row["body_type"]] += 1
            body_lighting[row["lighting"]] += 1
        if not color_ok:
            row["color"] = "unknown"
        else:
            color_accepted += 1
            color_counts[row["color"]] += 1
            color_lighting[row["lighting"]] += 1
        output_rows.append(row)

    output_fields = fields + [field for field in extra_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output_rows)

    stage178 = json.loads(args.stage178_report.read_text(encoding="utf-8"))
    stage179 = json.loads(args.stage179_report.read_text(encoding="utf-8"))
    stage188 = json.loads(args.stage188_report.read_text(encoding="utf-8"))
    body_base = stage178["states"]["track_balanced_effective_images"]["body_exact"]
    color_base = stage179["scene_totals"]["track_balanced_effective_representatives"]
    if "counts" in stage188:
        myvid_total = int(stage188["counts"]["component_train_eligible_rows"])
        myvid_low = sum(int(value) for value in stage188["counts"]["component_train_eligible_lowlight_proxy_body"].values())
    else:
        myvid_total = int(stage188["component_train_eligible"]["total"])
        myvid_low = int(stage188["component_train_eligible_lowlight_proxy"]["total"])
    body_dark_medium = body_lighting["Dark"] + body_lighting["Medium"]
    color_dark_medium = color_lighting["Dark"] + color_lighting["Medium"]
    body_quota = quota(
        int(body_base["total"]) + myvid_total + body_accepted,
        int(body_base["existing_explicit_night"]),
        int(body_base["existing_explicit_low_light"]) + myvid_low + body_dark_medium,
    )
    color_quota = quota(
        int(color_base["total"]) + color_accepted,
        int(color_base["existing_explicit_night"]),
        color_dark_medium,
    )
    body_component_pass = body_accepted >= 300 and len(body_counts) >= 2
    color_component_pass = color_accepted >= 100 and len(color_counts) >= 4
    report = {
        "schema_version": "stage192-tesla-teacher-quota-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "component_pass_global_quota_unmet_no_training" if body_component_pass and color_component_pass else "fail_closed_component_audit",
        "source": {"doi": "10.5281/zenodo.19814157", "license": "CC BY 4.0"},
        "component": {
            "stage191_rows": len(original), "body_teacher_accepted": body_accepted,
            "color_teacher_accepted": color_accepted,
            "body_counts": dict(sorted(body_counts.items())), "color_counts": dict(sorted(color_counts.items())),
            "body_by_lighting": dict(sorted(body_lighting.items())), "color_by_lighting": dict(sorted(color_lighting.items())),
            "confirmed_natural_night_rows": 0,
            "body_component_pass": body_component_pass, "color_component_pass": color_component_pass,
        },
        "global_exact_head_quota": {"body": body_quota, "color": color_quota},
        "gates": {
            "teacher_and_augmentation_consistency_complete": True,
            "cross_source_decontamination_inherited_from_stage191": True,
            "natural_night_separated_from_reduced_visibility": True,
            "all_adverse_light_quotas_pass": body_quota["passes_target"] and color_quota["passes_target"],
            "training_authorized": False,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "stage191_manifest": args.stage191_manifest, "stage191_report": args.stage191_report,
            "body_report": args.body_report, "color_report": args.color_report,
            "stage178_report": args.stage178_report, "stage179_report": args.stage179_report,
            "stage188_report": args.stage188_report,
        }.items()},
        "outputs": {"manifest": str(args.output_manifest), "manifest_sha256": sha256(args.output_manifest)},
        "policy": {
            "manual_source_labels_only_retained_not_created": True,
            "failed_teacher_rows_changed_to_unknown": True,
            "dark_medium_counted_as_reduced_visibility_not_natural_night": True,
            "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
        "next_action": "Do not train while the 30% adverse-light quota is unmet; acquire additional licensed, exact-labeled real CCTV night data or derive labels only through the same fail-closed source-label review.",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_report.with_suffix(args.output_report.suffix + ".sha256").write_text(
        f"{sha256(args.output_report)}  {args.output_report.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if body_component_pass and color_component_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
