"""Assemble an auditable stage-4 candidate gate report.

This report is intentionally fail-closed: a candidate is deployable only when
all independent precision/coverage, hard-scene improvement, parity, and TRT
smoke gates are true.  The frozen 60s video is never read by this script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(report: dict, head: str, key: str, default=None):
    return report.get(head, {}).get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = args.stage4_root / "baseline-hard-test.json"
    baseline = read_json(baseline_path)
    mining = read_json(args.mining_report)

    baseline_small = metric(
        baseline["stratified"], "body_type", "vehicle_size=small",  # pragma: no cover
    )
    # The helper above is for head-level reports; stratified is keyed by group.
    baseline_small = baseline["stratified"]["body_type"]["vehicle_size=small"]
    baseline_color = baseline["color"]
    candidate_names = [
        "ATTR-V2-MNV3-224-HARD",
        "ATTR-V2-MNV3-256-HARD",
        "ATTR-V2-EFFV2S-256-HARD",
        "ATTR-V2-BODYFOCUS-MNV3-256",
        "ATTR-V2-MNV3-256-HARD5",
    ]

    candidates = []
    for name in candidate_names:
        directory = args.stage4_root / name
        calibration = read_json(directory / "calibration-v2.json")
        hard_path = directory / "hard-test-stratified.json"
        hard = read_json(hard_path) if hard_path.exists() else None
        parity_path = directory / "onnx-parity.json"
        parity = read_json(parity_path) if parity_path.exists() else {"gate": False}
        smoke_log = directory / "trt-smoke.stdout.log"
        smoke_passed = smoke_log.exists() and "PASSED TensorRT.trtexec" in smoke_log.read_text(
            encoding="utf-8", errors="replace"
        )
        body_test = calibration["body_type"]["test"]
        color_test = calibration["color"]["test"]
        small = hard["stratified"]["body_type"].get("vehicle_size=small", {}) if hard else {}
        small_delta = small.get("high_confidence_coverage", 0.0) - baseline_small.get(
            "high_confidence_coverage", 0.0
        )
        candidate_color_unknown = hard["color"].get("predicted_unknown_rate", 0.0) if hard else None
        baseline_color_unknown = baseline_color.get("predicted_unknown_rate", 0.0)
        color_unknown_reduction = (
            (baseline_color_unknown - candidate_color_unknown) / baseline_color_unknown
            if hard and baseline_color_unknown > 0
            else None
        )
        body_precision_gate = body_test["precision"] >= 0.93
        color_precision_gate = color_test["precision"] >= 0.93
        body_coverage_gate = body_test["coverage"] >= 0.45
        color_coverage_gate = color_test["coverage"] >= 0.25
        # The hard-scene improvement gate is evaluated on the pre-declared
        # small-target slice.  It requires +15 percentage points, not merely
        # a precision/coverage trade-off on the aggregate test set.
        hard_improvement_gate = small_delta >= 0.15
        candidate = {
            "name": name,
            "artifacts": {"directory": str(directory), "checkpoint": str(directory / "best.pt"),
                           "onnx": str(directory / "calibrated.onnx"), "engine": str(directory / "calibrated.engine")},
            "calibrated_test": {"body_type": body_test, "color": color_test},
            "hard_test": {"body_type": hard["body_type"] if hard else None, "color": hard["color"] if hard else None,
                          "small_target_body": small,
                          "color_unknown_rate": candidate_color_unknown,
                          "color_unknown_reduction": color_unknown_reduction},
            "gates": {
                "body_precision": body_precision_gate,
                "body_coverage": body_coverage_gate,
                "color_precision": color_precision_gate,
                "color_coverage": color_coverage_gate,
                "small_target_coverage_plus_15pp": hard is not None and hard_improvement_gate,
                "complex_color_unknown_minus_20pct": hard is not None and color_unknown_reduction is not None and color_unknown_reduction >= 0.20,
                "onnx_pytorch_top1": bool(parity.get("gate", False)) and parity.get("body_top1_match_rate", 0.0) >= 0.995 and parity.get("color_top1_match_rate", 0.0) >= 0.995,
                "tensorrt_smoke": smoke_passed,
                # Candidate-specific runtime/trajectory evidence is required
                # before deployment.  It is deliberately false until the
                # candidate is wired into the isolated C++ harness and track
                # fusion replay.
                "trajectory_stability": False,
                "cxx_contract": False,
                "real_engine_smoke": False,
            },
        }
        candidate["hard_test"]["small_target_coverage_delta"] = small_delta if hard else None
        candidate["release_eligible"] = all(candidate["gates"].values())
        candidate["rejection_reasons"] = [key for key, passed in candidate["gates"].items() if not passed]
        candidates.append(candidate)

    result = {
        "schema_version": "stage4-hard-candidate-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_baseline": "vehicle-attr-agent-e-224",
        "frozen_video_used": False,
        "frozen_video_policy": "reserved for final replay only after every independent gate passes",
        "hard_mining": {"report": str(args.mining_report), "rows": mining.get("rows"),
                        "missing_or_unreadable": mining.get("missing_or_unreadable"),
                        "frozen_video_used": mining.get("frozen_video_used", False),
                        "tag_counts": mining.get("tag_counts", {}),
                        "score_bins": mining.get("score_bins", {})},
        "baseline_hard_test": {"report": str(baseline_path), "body_type": baseline["body_type"],
                               "color": baseline["color"],
                               "small_target_body_coverage": baseline_small.get("high_confidence_coverage"),
                               "color_unknown_rate": baseline_color.get("predicted_unknown_rate")},
        "candidates": candidates,
        "backend_validation": {
            "production_registry_contract": "pass",
            "registry_validator": "pass",
            "tensorrt_source_contract": "pass",
            "vehicle_cascade_contract_test": "pass",
            "dataset_manifest_contract_test": "pass",
            "vehicle_contract_test": "fail",
            "vehicle_contract_test_failure": "event attribute version must match runtime model selection",
            "candidate_track_fusion": "not_run_fail_closed",
            "candidate_cxx_contract": "not_run_fail_closed",
            "candidate_real_engine_smoke": "not_run_fail_closed_for_rejected_candidates",
        },
        "deployment": {"approved": False, "production_unchanged": True,
                        "reason": "No candidate meets the independent complex-scene +15pp coverage gate and all release gates.",
                        "rollback": "vehicle-attr-agent-e-224 remains the active rollback and production model."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
