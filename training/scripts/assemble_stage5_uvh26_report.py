#!/usr/bin/env python3
"""Create a fail-closed stage-5 candidate decision report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--test-metrics", type=Path, required=True)
    ap.add_argument("--uvh26-report", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate", default="ATTR-V3-MNV3-256-UVH26-BODY")
    args = ap.parse_args()
    cal = read(args.calibration)
    test = read(args.test_metrics)
    uvh = read(args.uvh26_report)
    body = cal["body_type"]
    color = cal["color"]
    gates = {
        "body_precision": body["test"]["precision"] >= 0.93,
        "body_coverage": body["test"]["coverage"] >= 0.45,
        "color_precision": color["test"]["precision"] >= 0.93,
        "color_coverage": color["test"]["coverage"] >= 0.25,
        "independent_release_target": bool(cal.get("release_target_met")),
        "hard_scene_improvement": False,
        "onnx_parity": False,
        "tensorrt": False,
        "track_stability": False,
        "frozen_video_not_used": True,
    }
    report = {
        "schema_version": "stage5-uvh26-candidate-report-v1",
        "candidate": args.candidate,
        "status": "rejected_fail_closed" if not all(gates.values()) else "eligible_pending_backend_gates",
        "production_model_unchanged": True,
        "production_artifact": "vehicle-attr-agent-e-224",
        "data": {"uvh26": uvh, "frozen_video_used": False},
        "calibrated": {"body_type": body, "color": color},
        "test_metrics": test,
        "gates": gates,
        "failure_reasons": [k for k, v in gates.items() if not v],
        "artifacts": {
            "checkpoint": {"path": str(args.checkpoint), "sha256": sha(args.checkpoint)},
            "calibration": {"path": str(args.calibration), "sha256": sha(args.calibration)},
            "test_metrics": {"path": str(args.test_metrics), "sha256": sha(args.test_metrics)},
            "uvh26_report": {"path": str(args.uvh26_report), "sha256": sha(args.uvh26_report)},
        },
        "next_action": "retain production baseline; tune color supervision/augmentation and retrain a new isolated candidate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
