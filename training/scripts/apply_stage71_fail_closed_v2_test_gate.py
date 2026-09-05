#!/usr/bin/env python3
"""Apply fixed fail-closed v2 track gates to the one-time Stage71 test.

The selected pair and every temporal parameter are read from the immutable v2
validation report. No test threshold, window, share, or margin search occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def number(value: Any, default: float = 0.0) -> float:
    return float(default if value is None else value)


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"fixed fail-closed v2 test evaluation failed ({result.returncode}); see {log}"
        )


def compact_fixed(
    report: dict[str, Any], head: str, window: int
) -> dict[str, Any]:
    fused = report.get(head, {}).get("fusion", {}).get(str(window))
    if not fused:
        return {
            "selection_status": "missing_fixed_window",
            "precision": 0.0,
            "coverage": 0.0,
            "effective_unknown_rate": 1.0,
            "weighted_stability_rate": 0.0,
        }
    final = fused.get("final_window", {})
    stability = fused.get("stability", {})
    return {
        "selection_status": "fixed_validation_parameters",
        "evaluated": final.get("evaluated"),
        "selected": final.get("selected"),
        "precision": number(final.get("precision")),
        "coverage": number(final.get("coverage")),
        "effective_unknown_rate": number(final.get("effective_unknown_rate"), 1.0),
        "weighted_stability_rate": number(stability.get("weighted_stability_rate")),
        "label_switches_total": int(stability.get("label_switches_total") or 0),
        "abstention_rate": number(stability.get("abstention_rate"), 1.0),
    }


def test_gates(
    body_vfg: dict[str, Any],
    color_vfg: dict[str, Any],
    body_ua: dict[str, Any],
    production_body_vfg: dict[str, Any],
    production_color_vfg: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    body_gain = number(body_vfg.get("coverage")) - number(
        production_body_vfg.get("coverage")
    )
    baseline_unknown = number(
        production_color_vfg.get("effective_unknown_rate"), 1.0
    )
    candidate_unknown = number(color_vfg.get("effective_unknown_rate"), 1.0)
    unknown_reduction = (
        (baseline_unknown - candidate_unknown) / baseline_unknown
        if baseline_unknown > 0.0
        else 0.0
    )
    gates = {
        "v2_test_vfg_body_precision_gte_0_93": number(body_vfg.get("precision")) >= 0.93,
        "v2_test_vfg_body_coverage_gte_0_45": number(body_vfg.get("coverage")) >= 0.45,
        "v2_test_vfg_color_precision_gte_0_93": number(color_vfg.get("precision")) >= 0.93,
        "v2_test_vfg_color_coverage_gte_0_25": number(color_vfg.get("coverage")) >= 0.25,
        "v2_test_ua_body_precision_gte_0_93": number(body_ua.get("precision")) >= 0.93,
        "v2_test_ua_body_coverage_gte_0_45": number(body_ua.get("coverage")) >= 0.45,
        "v2_test_vfg_body_stability_gte_0_95": number(
            body_vfg.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_test_vfg_color_stability_gte_0_95": number(
            color_vfg.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_test_ua_body_stability_gte_0_95": number(
            body_ua.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_test_body_coverage_gain_gte_15pp": body_gain >= 0.15,
        "v2_test_color_unknown_reduction_gte_20pct": unknown_reduction >= 0.20,
    }
    return gates, {
        "body_coverage_gain_percentage_points": body_gain * 100.0,
        "color_unknown_relative_reduction": unknown_reduction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-state", type=Path, required=True)
    parser.add_argument("--validation-v2-state", type=Path, required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--expected-vfg-manifest-sha256", required=True)
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-production-checkpoint-sha256", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--expected-v1-evaluator-sha256", required=True)
    parser.add_argument("--expected-v2-fusion-sha256", required=True)
    parser.add_argument("--expected-v2-fixed-evaluator-sha256", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pins = {
        "ua_manifest": (args.ua_manifest, args.expected_ua_manifest_sha256),
        "vfg_manifest": (args.vfg_manifest, args.expected_vfg_manifest_sha256),
        "production_checkpoint": (
            args.production_checkpoint,
            args.expected_production_checkpoint_sha256,
        ),
        "v1_evaluator": (
            args.scripts_root / "evaluate_attribute_track_fusion.py",
            args.expected_v1_evaluator_sha256,
        ),
        "v2_fusion": (
            args.scripts_root / "track_fusion_fail_closed_v2.py",
            args.expected_v2_fusion_sha256,
        ),
        "v2_fixed_evaluator": (
            args.scripts_root / "evaluate_attribute_track_fusion_fail_closed_v2.py",
            args.expected_v2_fixed_evaluator_sha256,
        ),
    }
    for label, (path, expected) in pins.items():
        require_sha(path, expected, label)

    final_state = json.loads(args.final_state.read_text(encoding="utf-8"))
    if final_state.get("status") != "pass_backend_eligible":
        raise RuntimeError("v1 final test is not backend eligible")
    final_report_path = Path(str(final_state.get("report", "")))
    final_report_sha = str(final_state.get("report_sha256", ""))
    require_sha(final_report_path, final_report_sha, "v1 final test report")
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    policy = final_report.get("policy", {})
    if policy.get("test_accessed") is not True or policy.get("test_used_for_selection") is not False:
        raise RuntimeError("v1 final test policy is invalid")
    if policy.get("frozen_video_used") is not False:
        raise RuntimeError("v1 final test accessed frozen video")

    validation_state = json.loads(args.validation_v2_state.read_text(encoding="utf-8"))
    if validation_state.get("status") != "complete":
        raise RuntimeError("v2 validation state is not complete")
    validation_report_path = Path(str(validation_state.get("report", "")))
    require_sha(
        validation_report_path,
        str(validation_state.get("report_sha256", "")),
        "v2 validation report",
    )
    validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
    selected_pair_id = final_report["selection"]["pair_id"]
    if selected_pair_id not in validation.get("passing_pairs", []):
        raise RuntimeError("v1 final test pair did not pass v2 validation")
    pair = next(
        item for item in validation["pairs"] if item.get("pair_id") == selected_pair_id
    )
    if pair.get("screen_status") != "pass":
        raise RuntimeError("selected pair is not v2 validation-passing")

    output_root = final_report_path.parent / "fail-closed-v2-test"
    if output_root.exists():
        raise FileExistsError("refusing to overwrite fail-closed v2 test evidence")
    output_root.mkdir(parents=False, exist_ok=False)

    def evaluate(
        dataset: str,
        manifest: Path,
        model_name: str,
        checkpoint: Path,
        checkpoint_sha: str,
        head: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        require_sha(checkpoint, checkpoint_sha, f"{model_name} checkpoint")
        threshold = number(settings["threshold"])
        window = int(settings["window"])
        share = number(settings["minimum_share"])
        margin = number(settings["minimum_margin"])
        slug = safe_name(f"{dataset}-{model_name}-{head}")
        output = output_root / f"{slug}.json"
        log = output_root / f"{slug}.log"
        run(
            [
                args.python,
                str(pins["v2_fixed_evaluator"][0]),
                "--manifest", str(manifest),
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--split", "test",
                "--type-threshold", str(threshold),
                "--color-threshold", str(threshold),
                "--fusion-windows", str(window),
                "--fusion-min-share", str(share),
                "--fusion-min-margin", str(margin),
                "--minimum-window-frames", "3",
                "--batch-size", "128",
                "--workers", "4",
                "--device", args.device,
            ],
            log,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("schema_version") != "attribute-track-fusion-fixed-fail-closed-v2":
            raise RuntimeError("fixed evaluator did not emit v2 evidence")
        if report.get("protocol", {}).get("parameter_search") is not False:
            raise RuntimeError("fixed evaluator searched test parameters")
        result = compact_fixed(report, head, window)
        result.update(
            {
                "threshold": threshold,
                "window": window,
                "minimum_share": share,
                "minimum_margin": margin,
                "report": str(output.resolve()),
                "report_sha256": sha256(output),
                "checkpoint_sha256": checkpoint_sha.lower(),
            }
        )
        return result

    production = validation["production_fail_closed_v2"]
    production_sha = args.expected_production_checkpoint_sha256
    production_body_vfg = evaluate(
        "vfg", args.vfg_manifest, "production-body", args.production_checkpoint,
        production_sha, "body_type", production["body_vfg"],
    )
    production_color_vfg = evaluate(
        "vfg", args.vfg_manifest, "production-color", args.production_checkpoint,
        production_sha, "color", production["color_vfg"],
    )
    production_body_ua = evaluate(
        "ua", args.ua_manifest, "production-body", args.production_checkpoint,
        production_sha, "body_type", production["body_ua"],
    )
    body_checkpoint = Path(pair["body_checkpoint"])
    color_checkpoint = Path(pair["color_checkpoint"])
    body_vfg = evaluate(
        "vfg", args.vfg_manifest, pair["body_model"], body_checkpoint,
        pair["body_checkpoint_sha256"], "body_type", pair["fail_closed_v2"]["body_vfg"],
    )
    color_vfg = evaluate(
        "vfg", args.vfg_manifest, pair["color_model"], color_checkpoint,
        pair["color_checkpoint_sha256"], "color", pair["fail_closed_v2"]["color_vfg"],
    )
    body_ua = evaluate(
        "ua", args.ua_manifest, pair["body_model"], body_checkpoint,
        pair["body_checkpoint_sha256"], "body_type", pair["fail_closed_v2"]["body_ua"],
    )

    gates, comparison = test_gates(
        body_vfg,
        color_vfg,
        body_ua,
        production_body_vfg,
        production_color_vfg,
    )
    backup = final_report_path.parent / "final-test-report-v1-pre-fail-closed-v2.json"
    if backup.exists():
        raise FileExistsError("refusing to overwrite v1 final test backup")
    shutil.copy2(final_report_path, backup)
    require_sha(backup, final_report_sha, "v1 final test backup")

    final_report["fail_closed_v2_test"] = {
        "parameters_source": str(validation_report_path.resolve()),
        "parameters_source_sha256": sha256(validation_report_path),
        "selected_pair": selected_pair_id,
        "production": {
            "body_vfg": production_body_vfg,
            "color_vfg": production_color_vfg,
            "body_ua": production_body_ua,
        },
        "candidate": {
            "body_vfg": body_vfg,
            "color_vfg": color_vfg,
            "body_ua": body_ua,
        },
        "comparison_to_production": comparison,
        "gates": gates,
        "test_parameter_search": False,
        "frozen_video_used": False,
    }
    final_report["gates"].update(gates)
    passed = all(final_report["gates"].values())
    final_report["status"] = (
        "pass_backend_eligible" if passed
        else "fail_closed_before_backend"
    )
    final_report["policy"]["fail_closed_v2_test_applied"] = True
    final_report["policy"]["fail_closed_v2_test_parameter_search"] = False
    final_report["evidence"]["v1_final_report_backup"] = {
        "path": str(backup.resolve()),
        "sha256": sha256(backup),
    }
    final_report["next_action"] = (
        "export isolated candidate ONNX and run backend gates"
        if passed
        else "reject Stage71 pair before ONNX/backend/frozen replay/deployment"
    )
    atomic_json(final_report_path, final_report)
    final_state.update(
        {
            "status": final_report["status"],
            "updated_at": now(),
            "report_sha256": sha256(final_report_path),
            "fail_closed_v2_test_applied": True,
            "fail_closed_v2_test_parameter_search": False,
            "backend_eligible": passed,
        }
    )
    atomic_json(args.final_state, final_state)
    print(
        json.dumps(
            {
                "status": final_report["status"],
                "selected_pair": selected_pair_id,
                "gates": gates,
                "report": str(final_report_path.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
