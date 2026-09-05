#!/usr/bin/env python3
"""Summarize two Stage104 validation reports without accessing test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


COLOR_GATES = (
    "color_static_precision",
    "color_static_coverage",
    "color_complex_unknown_reduction",
    "color_track_precision",
    "color_track_coverage",
    "color_track_stability",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(reports_dir: Path) -> dict[str, object]:
    variants: dict[str, object] = {}
    qualified: list[str] = []
    for name in ("best", "gate-best"):
        path = reports_dir / f"{name}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete_validation_only":
            raise ValueError(f"invalid validation status: {path}")
        policy = report.get("policy", {})
        if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
            raise ValueError(f"unsafe validation policy: {path}")
        gates = report["shared_validation_gates"]["gates"]
        color_gates = {key: bool(gates[key]) for key in COLOR_GATES}
        all_pass = all(color_gates.values())
        if all_pass:
            qualified.append(name)
        variants[name] = {
            "static": report["threshold_selection"]["color_shared"],
            "complex_unknown_relative_reduction": report["comparison"][
                "color_complex_static_unknown_relative_reduction"
            ],
            "track_final": report["track_fusion"]["candidate"]["color_shared"]["track_final"],
            "stability": report["track_fusion"]["candidate"]["color_shared"]["stability"],
            "color_gates": color_gates,
            "all_color_gates_pass": all_pass,
            "report_sha256": sha256(path),
        }
    return {
        "status": "complete_validation_only",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "decision": "color_component_qualified" if qualified else "color_candidate_rejected_fail_closed",
        "qualified_variants": qualified,
        "variants": variants,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reports_dir.resolve())
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_state.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
