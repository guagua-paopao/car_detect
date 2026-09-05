#!/usr/bin/env python3
"""Decide whether the type-only Stage67 UVH-26 follow-up is justified.

The decision is intentionally conservative: Stage67 is skipped when any Stage66
candidate already satisfies the complete type-side validation screen, because
UVH-26 adds exact body labels but no exact color labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


TYPE_GATES = (
    "hard_body_precision_0_93",
    "hard_body_coverage_0_45",
    "hard_body_coverage_gain_15pp",
    "ua_body_precision_0_93",
    "ua_body_coverage_0_45",
    "ua_body_stability_0_95",
)
VFG_TYPE_GATES = ("static_body", "track_body", "type_coverage_gain_15pp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_isolation(state: dict, report: dict, report_path: Path) -> None:
    if state.get("status") != "complete":
        raise RuntimeError("Stage66 validation state is not complete")
    if state.get("evaluator_policy_revision") != "attribute-truth-routed-v2":
        raise RuntimeError("Stage66 validation did not use attribute-truth-routed-v2")
    if Path(state.get("report", "")) != report_path.resolve():
        raise RuntimeError("Stage66 validation state does not pin the supplied report")
    if str(state.get("report_sha256", "")).lower() != sha256(report_path):
        raise RuntimeError("Stage66 validation report hash mismatch")

    policy = report.get("policy", {})
    if policy.get("validation_only") is not True:
        raise RuntimeError("Stage66 report is not validation-only")
    for key in (
        "test_accessed",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage66 validation isolation violation: {key}")
    if policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage66 report lost the deployment pause")
    routing = policy.get("attribute_truth_routing", {})
    if "body" not in str(routing.get("ua_detrac", "")).lower():
        raise RuntimeError("Stage66 report lost UA body truth routing")
    if "body/color" not in str(routing.get("vfg7", "")).lower():
        raise RuntimeError("Stage66 report lost VFG labeled body/color routing")


def summarize_type_candidate(item: dict) -> dict:
    gates = item.get("gates", {})
    vfg = item.get("vfg7", {}).get("validation_gates", {})
    required = {key: gates.get(key) is True for key in TYPE_GATES}
    required.update({f"vfg_{key}": vfg.get(key) is True for key in VFG_TYPE_GATES})
    hard = item.get("hard", {}).get("body_type", {})
    ua = item.get("ua_track", {}).get("body_type", {})
    vfg_threshold = item.get("vfg7", {}).get("selected_thresholds", {}).get("body_type", {})
    return {
        "model": item.get("model"),
        "checkpoint": item.get("checkpoint"),
        "checkpoint_sha256": item.get("checkpoint_sha256"),
        "type_gate_pass": all(required.values()),
        "passed_type_gate_count": sum(required.values()),
        "required_type_gate_count": len(required),
        "type_gates": required,
        "hard_body_precision": hard.get("precision"),
        "hard_body_coverage": hard.get("coverage"),
        "ua_body_precision": ua.get("precision"),
        "ua_body_coverage": ua.get("coverage"),
        "ua_body_stability": ua.get("weighted_stability_rate"),
        "vfg_body_precision": vfg_threshold.get("precision"),
        "vfg_body_coverage": vfg_threshold.get("coverage"),
    }


def decision_from_report(report: dict) -> dict:
    candidates = [
        summarize_type_candidate(item)
        for item in report.get("models", [])
        if item.get("model") != "production"
    ]
    if not candidates:
        raise RuntimeError("Stage66 report contains no candidate models")

    type_passing = [item for item in candidates if item["type_gate_pass"]]
    full_passing = list(report.get("passing_candidates", []))
    if full_passing:
        action = "skip_stage67_stage66_full_candidate_available"
        reason = "Stage66 already has a candidate that passes the complete validation screen"
    elif type_passing:
        action = "skip_stage67_type_gates_met_focus_next_round_on_color"
        reason = "at least one Stage66 candidate passes every type-side gate; type-only UVH-26 is not the limiting evidence"
    else:
        action = "run_stage67_uvh26_controlled_type_followup"
        reason = "no Stage66 candidate passes every complex-scene type gate"

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["passed_type_gate_count"],
            float(item.get("hard_body_coverage") or 0.0),
            float(item.get("ua_body_coverage") or 0.0),
            float(item.get("vfg_body_coverage") or 0.0),
        ),
        reverse=True,
    )
    return {
        "action": action,
        "reason": reason,
        "full_passing_candidates": full_passing,
        "type_passing_candidates": [item["model"] for item in type_passing],
        "recommended_stage67_initialization_source": ranked[0],
        "candidate_type_summaries": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-state", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage67 decision evidence")
    state = json.loads(args.validation_state.read_text(encoding="utf-8"))
    report_path = args.validation_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_isolation(state, report, report_path)
    decision = decision_from_report(report)
    payload = {
        "schema_version": "attribute-stage67-uvh26-followup-decision-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage66_validation_state": str(args.validation_state.resolve()),
        "stage66_validation_state_sha256": sha256(args.validation_state),
        "stage66_validation_report": str(report_path),
        "stage66_validation_report_sha256": sha256(report_path),
        **decision,
        "policy": {
            "uvh26_exact_body_only": True,
            "uvh26_exact_color_rows": 0,
            "do_not_run_when_type_gates_already_pass": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "pass", "action": payload["action"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
