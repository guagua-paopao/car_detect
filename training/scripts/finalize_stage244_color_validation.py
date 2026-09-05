#!/usr/bin/env python3
"""Apply Stage244's stricter no-regression gates to Stage159 color reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--reference-state", required=True, type=Path)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--training-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if sha256(args.reference_state).lower() != args.expected_reference_sha256.lower():
        raise RuntimeError("reference Stage159 state SHA256 mismatch")
    reference_state = json.loads(args.reference_state.read_text(encoding="utf-8"))
    reference = reference_state.get("selected_color")
    if not reference or not reference.get("qualified"):
        raise RuntimeError("Stage159 reference color is not qualified")
    training_state = json.loads(args.training_state.read_text(encoding="utf-8"))
    if training_state.get("status") != "training_complete_pending_stage244_validation":
        raise RuntimeError("Stage243 training state is not ready")

    reports = sorted(args.validation_root.glob("*/report.json"))
    if len(reports) != 2:
        raise RuntimeError(f"expected two Stage244 reports, got {len(reports)}")
    evaluated = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        policy = report.get("policy", {})
        if report.get("head") != "color" or policy.get("split") != "validation":
            raise RuntimeError(f"invalid validation report: {path}")
        if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
            raise RuntimeError(f"validation isolation failed: {path}")
        overall = report["static"]["candidate_overall"]
        lighting = report["stratified"]["lighting"]
        sizes = report["stratified"]["vehicle_size"]
        per_class = report["per_class"]
        ref_overall = reference["overall"]
        ref_lighting = reference["stratified"]["lighting"]
        ref_sizes = reference["stratified"]["vehicle_size"]
        ref_classes = reference["per_class"]
        gates = {
            "stage159_core_color_gates": bool(report["gates"]["all_pass"]),
            "overall_precision_at_least_0_93": overall["precision"] >= 0.93,
            "overall_coverage_no_more_than_0_02_regression": overall["coverage"] >= ref_overall["coverage"] - 0.02,
            "daylight_precision_no_more_than_0_005_regression": lighting["daylight"]["precision"] >= ref_lighting["daylight"]["precision"] - 0.005,
            "daylight_coverage_no_more_than_0_03_regression": lighting["daylight"]["coverage"] >= ref_lighting["daylight"]["coverage"] - 0.03,
            "night_precision_at_least_0_90": lighting["night"]["precision"] >= 0.90,
            "night_precision_improves_reference": lighting["night"]["precision"] > ref_lighting["night"]["precision"],
            "night_coverage_no_more_than_0_10_regression": lighting["night"]["coverage"] >= ref_lighting["night"]["coverage"] - 0.10,
            "low_light_precision_at_least_0_84": lighting["low_light"]["precision"] >= 0.84,
            "low_light_precision_improves_reference": lighting["low_light"]["precision"] > ref_lighting["low_light"]["precision"],
            "low_light_coverage_no_more_than_0_08_regression": lighting["low_light"]["coverage"] >= ref_lighting["low_light"]["coverage"] - 0.08,
            "small_precision_at_least_0_91": sizes["small"]["precision"] >= 0.91,
            "small_coverage_no_more_than_0_05_regression": sizes["small"]["coverage"] >= ref_sizes["small"]["coverage"] - 0.05,
            "black_precision_no_regression": per_class["black"]["precision"] >= ref_classes["black"]["precision"],
        }
        for color in ("blue", "red", "silver", "white"):
            gates[f"{color}_precision_no_more_than_0_02_regression"] = (
                per_class[color]["precision"] >= ref_classes[color]["precision"] - 0.02
            )
        qualified = all(gates.values())
        improvement_score = (
            3.0 * (lighting["night"]["precision"] - ref_lighting["night"]["precision"])
            + 2.0 * (lighting["low_light"]["precision"] - ref_lighting["low_light"]["precision"])
            + (lighting["night"]["coverage"] - ref_lighting["night"]["coverage"])
            + (lighting["low_light"]["coverage"] - ref_lighting["low_light"]["coverage"])
            + 0.5 * (overall["coverage"] - ref_overall["coverage"])
        )
        evaluated.append(
            {
                "variant": path.parent.name,
                "qualified": qualified,
                "checkpoint": report["inputs"]["candidate_checkpoint"],
                "checkpoint_sha256": report["inputs"]["candidate_checkpoint_sha256"],
                "thresholds": report["selection"]["thresholds"],
                "overall": overall,
                "complex": report["static"]["candidate_complex"],
                "comparison": report["comparison"],
                "track": report["track_fusion"],
                "lighting": lighting,
                "vehicle_size": sizes,
                "per_class": per_class,
                "gates": gates,
                "improvement_score": improvement_score,
                "report": str(path),
                "report_sha256": sha256(path),
            }
        )
    passed = [item for item in evaluated if item["qualified"]]
    selected = max(passed, key=lambda item: (item["improvement_score"], item["overall"]["coverage"])) if passed else None
    result = {
        "schema_version": "stage244-color-validation-state-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_candidate_selected" if selected else "complete_fail_closed_no_qualified_candidate",
        "reference": {
            "checkpoint": reference["checkpoint"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "overall": reference["overall"],
            "lighting": reference["stratified"]["lighting"],
            "vehicle_size": reference["stratified"]["vehicle_size"],
            "per_class": reference["per_class"],
            "state": str(args.reference_state),
            "state_sha256": sha256(args.reference_state),
        },
        "selected": selected,
        "candidates": evaluated,
        "research_only": True,
        "deployment_eligible": False,
        "next_gate": "new untouched holdout validation before ONNX/backend work" if selected else "reject Stage243 and retain Stage159 color candidate",
        "policy": {
            "thresholds_selected_on_validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256(args.output) + "  " + args.output.name + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": selected and selected["variant"], "qualified": [item["variant"] for item in passed]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
