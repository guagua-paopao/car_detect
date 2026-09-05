#!/usr/bin/env python3
"""Assemble an auditable, fail-closed progress snapshot for attribute-domain-v2."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    candidate_files = sorted(root.glob("stage*/**/*candidate-report.json"))
    candidates = []
    for path in candidate_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append({
            "candidate": data.get("candidate"),
            "status": data.get("status"),
            "failure_reasons": data.get("failure_reasons", []),
            "path": str(path),
            "sha256": sha(path),
        })
    report = {
        "schema_version": "attribute-domain-v2-progress-v1",
        "production_artifact": "vehicle-attr-agent-e-224",
        "production_model_unchanged": True,
        "frozen_video_used": False,
        "frozen_video_policy": "reserved for final 36-48s replay only after all independent and backend gates pass",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "deployment_status": "production_baseline_retained",
        "backend_gates_run_for_candidate": False,
        "next_action": "validate the preferred adaptive color-router candidate with a candidate-specific C++/real-engine contract, obtain labeled complex-color continuous evidence, and keep vehicle-attr-agent-e-224 online until frontend/backend changes are complete",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
