#!/usr/bin/env python3
"""Summarize validation-only truck subtype threshold sweep reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BODY_GATES = (
    "body_static_precision",
    "body_static_coverage",
    "body_complex_coverage_gain",
    "body_track_precision",
    "body_track_coverage",
    "body_track_stability",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(reports_dir: Path) -> dict[str, object]:
    variants: list[dict[str, object]] = []
    for path in sorted(reports_dir.glob("subtype-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete_validation_only":
            raise ValueError(f"invalid validation status: {path}")
        policy = report.get("policy", {})
        if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
            raise ValueError(f"unsafe validation policy: {path}")
        gates = report["shared_validation_gates"]["gates"]
        body_gates = {key: bool(gates[key]) for key in BODY_GATES}
        static = report["threshold_selection"]["body_exact"]
        track = report["track_fusion"]["candidate"]["body_exact"]
        item = {
            "variant": path.stem,
            "subtype_threshold": report["inputs"]["body_specialist_subtype_threshold"],
            "static": static,
            "complex_coverage_gain": report["comparison"]["body_complex_static_coverage_gain"],
            "track_final": track["track_final"],
            "stability": track["stability"],
            "body_gates": body_gates,
            "all_body_gates_pass": all(body_gates.values()),
            "report_sha256": sha256(path),
        }
        variants.append(item)
    if not variants:
        raise ValueError("no subtype sweep reports found")

    def rank(item: dict[str, object]) -> tuple[float, ...]:
        gates = item["body_gates"]
        static = item["static"]
        track = item["track_final"]
        precision_safe = gates["body_static_precision"] and gates["body_track_precision"]
        return (
            float(item["all_body_gates_pass"]),
            float(precision_safe),
            float(sum(gates.values())),
            float(item["complex_coverage_gain"]),
            float(static["coverage"]),
            float(track["coverage"]),
        )

    selected = max(variants, key=rank)
    qualified = [item["variant"] for item in variants if item["all_body_gates_pass"]]
    return {
        "status": "complete_validation_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "body_subtype_threshold_qualified" if qualified else "threshold_only_repair_rejected_fail_closed",
        "qualified_variants": qualified,
        "selected_diagnostic": selected["variant"],
        "selected_diagnostic_reason": "lexicographic body-gate count, complex coverage gain, static coverage and track coverage; never substitutes for all gates",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reports_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
