#!/usr/bin/env python3
"""Rebuild a Stage62 screen without treating absent UA color truth as failure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from run_stage62_domain_validation_after_training import build_release_gates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("schema_version") != "attribute-stage62-domain-validation-screen-v1":
        raise RuntimeError("unsupported source validation schema")
    corrected = copy.deepcopy(source)
    corrected["schema_version"] = "attribute-stage62-domain-validation-screen-v2"
    corrected["created_at"] = datetime.now(timezone.utc).isoformat()
    corrected["supersedes"] = {
        "report": str(args.input.resolve()),
        "sha256": sha256(args.input),
        "reason": "UA-DETRAC validation has zero supervised color rows; its zero-valued color metrics were not valid release evidence",
    }
    baseline = corrected["models"][0]
    passing: list[str] = []
    for item in corrected["models"]:
        hard_body = item["hard"]["body_type"]
        hard_color = item["hard"]["color"]
        ua_body = item["ua_track"]["body_type"]
        comparison = item["comparison_to_production"]
        gates = build_release_gates(
            hard_body=hard_body,
            hard_color=hard_color,
            ua_body=ua_body,
            vfg_screen_passed=item["vfg7"].get("validation_screen_status") == "pass",
            body_gain=float(comparison["hard_body_coverage_gain_percentage_points"]) / 100.0,
            color_unknown_reduction=float(comparison["hard_color_unknown_relative_reduction"]),
            is_baseline=item is baseline,
        )
        item["gates"] = gates
        item["ua_color_evidence_policy"] = {
            "release_gate": False,
            "reason": "zero supervised color rows in the UA-DETRAC validation manifest",
            "labeled_color_track_gate": "vfg_labeled_body_color_and_track_screen",
        }
        item["screen_status"] = "pass" if all(gates.values()) else "fail_closed"
        if item is not baseline and item["screen_status"] == "pass":
            passing.append(item["model"])
    corrected["passing_candidates"] = passing
    corrected["status"] = "pass_candidates_available" if passing else "complete_all_candidates_rejected"
    corrected["policy"]["attribute_truth_routing"] = {
        "ua_detrac": "body precision, coverage and stability only",
        "vfg7": "labeled body/color static and track precision, coverage and stability",
        "ua_color": "diagnostic only",
    }
    corrected["decision"] = (
        "passing candidates may proceed to seed robustness before a single final test"
        if passing
        else "reject all candidates before test/backend/frozen-video/deployment and continue with Stage63"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": corrected["status"], "passing_candidates": passing, "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
