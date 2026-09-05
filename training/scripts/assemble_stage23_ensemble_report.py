#!/usr/bin/env python3
"""Record Stage23 body + production-color ensemble with fail-closed gates."""

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
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensemble = json.loads(args.ensemble.read_text(encoding="utf-8"))
    validation = ensemble["validation"]
    test = ensemble["test"]
    gates = {
        "body_precision": test["body_type"]["high_confidence_precision"] >= 0.93,
        "body_coverage": test["body_type"]["high_confidence_coverage"] >= 0.45,
        "color_precision": test["color"]["high_confidence_precision"] >= 0.93,
        "color_coverage": test["color"]["high_confidence_coverage"] >= 0.25,
        # Production hard-v2 body coverage is 0.763 at the same direct protocol;
        # this candidate is 0.524, so the +15pp complex-coverage gate is false.
        "complex_type_coverage_plus_15pp": False,
        "complex_color_unknown_minus_20pct": False,
        "trajectory_stability": False,
        "onnx_pytorch_parity": False,
        "tensorrt_smoke": False,
    }
    report = {
        "schema_version": "attribute-ensemble-candidate-v2",
        "candidate": "ATTR-V23-BODY-BMDRAW-TYPE+ATTR-AGENT-E-224-COLOR",
        "status": "rejected_fail_closed",
        "production_artifact": "vehicle-attr-agent-e-224",
        "production_model_unchanged": True,
        "frozen_video_used": False,
        "manifest_report": json.loads(args.manifest_report.read_text(encoding="utf-8")),
        "ensemble": str(args.ensemble),
        "ensemble_sha256": sha(args.ensemble),
        "validation": validation,
        "test": test,
        "gates": gates,
        "failure_reasons": [key for key, passed in gates.items() if not passed],
        "next_action": "retain production baseline; improve complex-scene body coverage and validate track fusion before any deployment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
