#!/usr/bin/env python3
"""Fail-closed independent report for a ResNet50 attribute candidate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--baseline-hard", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate", default="ATTR-V12-RESNET50-256-HARD")
    args = ap.parse_args()
    cal, test, base = load(args.calibration), load(args.test), load(args.baseline_hard)
    body, color = cal["body_type"]["test"], cal["color"]["test"]
    hard_body = test["stratified"]["body_type"]
    small_base = base["stratified"]["body_type"]["vehicle_size=small"]["high_confidence_coverage"]
    small_candidate = hard_body["vehicle_size=small"]["high_confidence_coverage"]
    unknown_base = base["color"]["predicted_unknown_rate"]
    unknown_candidate = test["color"]["predicted_unknown_rate"]
    gates = {
        "body_precision": body["precision"] >= .93,
        "body_coverage": body["coverage"] >= .45,
        "color_precision": color["precision"] >= .93,
        "color_coverage": color["coverage"] >= .25,
        "complex_type_coverage_plus_15pp": small_candidate - small_base >= .15,
        "complex_color_unknown_minus_20pct": unknown_base > 0 and (unknown_base - unknown_candidate) / unknown_base >= .20,
        "trajectory_stability": False,
        "onnx_pytorch_parity": False,
        "tensorrt_smoke": False,
        "frozen_video_not_used": True,
    }
    report = {
        "schema_version": "attribute-resnet-candidate-report-v1",
        "candidate": args.candidate,
        "status": "eligible_pending_backend_gates" if all(gates.values()) else "rejected_fail_closed",
        "production_model_unchanged": True,
        "production_artifact": "vehicle-attr-agent-e-224",
        "frozen_video_used": False,
        "metrics": {"calibrated_test": {"body_type": body, "color": color}, "hard_stratified": hard_body},
        "complex_scene_comparison": {
            "baseline_small_type_coverage": small_base,
            "candidate_small_type_coverage": small_candidate,
            "delta": small_candidate - small_base,
            "baseline_color_unknown_rate": unknown_base,
            "candidate_color_unknown_rate": unknown_candidate,
        },
        "gates": gates,
        "failure_reasons": [key for key, value in gates.items() if not value],
        "artifacts": {key: {"path": str(path), "sha256": sha(path)} for key, path in {
            "calibration": args.calibration, "test": args.test, "baseline_hard": args.baseline_hard, "checkpoint": args.checkpoint
        }.items()},
        "next_action": "retain production baseline; continue isolated small-target/type-feature experiments" if not all(gates.values()) else "run ONNX/TRT/backend gates before deployment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
