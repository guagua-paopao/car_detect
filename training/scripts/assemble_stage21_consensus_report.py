#!/usr/bin/env python3
"""Record Stage21 consensus-body results as a fail-closed candidate report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    test = json.loads(args.test_report.read_text(encoding="utf-8"))
    body = test["body_type"]
    color = test["color"]
    gates = {
        "body_precision": body["high_confidence_precision"] >= 0.93,
        "body_coverage": body["high_confidence_coverage"] >= 0.45,
        "color_precision": color["high_confidence_precision"] >= 0.93,
        "color_coverage": color["high_confidence_coverage"] >= 0.25,
        "complex_type_coverage_plus_15pp": False,
        "complex_color_unknown_minus_20pct": False,
        "trajectory_stability": False,
        "onnx_pytorch_parity": False,
        "tensorrt_smoke": False,
    }
    report = {
        "schema_version": "attribute-consensus-body-candidate-v1",
        "candidate": "ATTR-V18-MNV3-256-BMDCONSENSUS-BODY",
        "status": "rejected_fail_closed",
        "production_artifact": "vehicle-attr-agent-e-224",
        "production_model_unchanged": True,
        "frozen_video_used": False,
        "manifest_report": json.loads(args.manifest_report.read_text(encoding="utf-8")),
        "test_report": str(args.test_report),
        "test_report_sha256": sha(args.test_report),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "gates": gates,
        "failure_reasons": [key for key, passed in gates.items() if not passed],
        "next_action": "retain production baseline; train strict BMD source-aligned type candidate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
