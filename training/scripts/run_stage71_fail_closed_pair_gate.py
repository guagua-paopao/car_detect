#!/usr/bin/env python3
"""Apply fail-closed v2 track fusion to Stage71 validation-passing pairs.

This is an additional validation-only gate. It consumes the immutable static
and v1 track screen, evaluates only its passing pairs with the stricter v2
temporal policy, and emits a compatible narrowed pair report. Test and frozen
video inputs are structurally rejected by the called sweeper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
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
            f"fail-closed track validation failed ({result.returncode}); see {log}"
        )


def number(value: Any, default: float = 0.0) -> float:
    return float(default if value is None else value)


def compact_selected(report: dict[str, Any], head: str) -> dict[str, Any]:
    selected = report.get(head, {}).get("selected")
    if not selected:
        return {
            "selection_status": "no_selection",
            "precision": 0.0,
            "coverage": 0.0,
            "effective_unknown_rate": 1.0,
            "weighted_stability_rate": 0.0,
            "label_switches_total": 0,
        }
    final = selected.get("final_window", {})
    stability = selected.get("stability", {})
    return {
        "selection_status": "selected",
        "threshold": selected.get("threshold"),
        "window": selected.get("window"),
        "minimum_share": selected.get("minimum_share"),
        "minimum_margin": selected.get("minimum_margin"),
        "evaluated": final.get("evaluated"),
        "selected": final.get("selected"),
        "precision": number(final.get("precision")),
        "coverage": number(final.get("coverage")),
        "effective_unknown_rate": number(final.get("effective_unknown_rate"), 1.0),
        "weighted_stability_rate": number(stability.get("weighted_stability_rate")),
        "label_switches_total": int(stability.get("label_switches_total") or 0),
        "abstention_rate": number(stability.get("abstention_rate"), 1.0),
    }


def pair_gates(
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
        "v2_vfg_body_precision_gte_0_93": number(body_vfg.get("precision")) >= 0.93,
        "v2_vfg_body_coverage_gte_0_45": number(body_vfg.get("coverage")) >= 0.45,
        "v2_vfg_color_precision_gte_0_93": number(color_vfg.get("precision")) >= 0.93,
        "v2_vfg_color_coverage_gte_0_25": number(color_vfg.get("coverage")) >= 0.25,
        "v2_ua_body_precision_gte_0_93": number(body_ua.get("precision")) >= 0.93,
        "v2_ua_body_coverage_gte_0_45": number(body_ua.get("coverage")) >= 0.45,
        "v2_vfg_body_stability_gte_0_95": number(
            body_vfg.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_vfg_color_stability_gte_0_95": number(
            color_vfg.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_ua_body_stability_gte_0_95": number(
            body_ua.get("weighted_stability_rate")
        ) >= 0.95,
        "v2_body_coverage_gain_gte_15pp": body_gain >= 0.15,
        "v2_color_unknown_reduction_gte_20pct": unknown_reduction >= 0.20,
    }
    return gates, {
        "body_coverage_gain_percentage_points": body_gain * 100.0,
        "color_unknown_relative_reduction": unknown_reduction,
    }


def validate_upstream_report(
    state_path: Path,
) -> tuple[Path, dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError("upstream student validation is not complete")
    report_path = Path(str(state.get("report", "")))
    require_sha(report_path, str(state.get("report_sha256", "")), "upstream pair report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = report.get("policy", {})
    if policy.get("validation_only") is not True:
        raise RuntimeError("upstream pair report is not validation-only")
    for key in (
        "test_accessed",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if policy.get(key) is not False:
            raise RuntimeError(f"upstream pair policy violation: {key}")
    if report.get("status") != "pass_pairs_available" or not report.get("passing_pairs"):
        raise RuntimeError("upstream validation has no passing pair")
    return report_path, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--expected-vfg-manifest-sha256", required=True)
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-production-checkpoint-sha256", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--expected-v1-sweeper-sha256", required=True)
    parser.add_argument("--expected-v1-evaluator-sha256", required=True)
    parser.add_argument("--expected-v2-fusion-sha256", required=True)
    parser.add_argument("--expected-v2-sweeper-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite fail-closed track evidence")
    pins = {
        "ua_manifest": (args.ua_manifest, args.expected_ua_manifest_sha256),
        "vfg_manifest": (args.vfg_manifest, args.expected_vfg_manifest_sha256),
        "production_checkpoint": (
            args.production_checkpoint,
            args.expected_production_checkpoint_sha256,
        ),
        "v1_sweeper": (
            args.scripts_root / "sweep_attribute_track_fusion.py",
            args.expected_v1_sweeper_sha256,
        ),
        "v1_evaluator": (
            args.scripts_root / "evaluate_attribute_track_fusion.py",
            args.expected_v1_evaluator_sha256,
        ),
        "v2_fusion": (
            args.scripts_root / "track_fusion_fail_closed_v2.py",
            args.expected_v2_fusion_sha256,
        ),
        "v2_sweeper": (
            args.scripts_root / "sweep_attribute_track_fusion_fail_closed_v2.py",
            args.expected_v2_sweeper_sha256,
        ),
    }
    for label, (path, expected) in pins.items():
        require_sha(path, expected, label)

    upstream_path, upstream = validate_upstream_report(args.upstream_state)
    upstream_pairs = {
        pair["pair_id"]: pair for pair in upstream.get("pairs", [])
    }
    passing_ids = list(upstream["passing_pairs"])
    if any(pair_id not in upstream_pairs for pair_id in passing_ids):
        raise RuntimeError("upstream passing-pair list lacks pair evidence")

    vfg_source = Path(upstream["artifacts"]["vfg_comparison"])
    require_sha(
        vfg_source,
        upstream["artifacts"]["vfg_comparison_sha256"],
        "upstream VFG comparison",
    )
    vfg_upstream = json.loads(vfg_source.read_text(encoding="utf-8"))
    production_vfg = next(
        item for item in vfg_upstream["models"] if item.get("model") == "production"
    )
    production_body_threshold = number(
        production_vfg["selected_thresholds"]["body_type"]["threshold"]
    )
    production_color_threshold = number(
        production_vfg["selected_thresholds"]["color"]["threshold"]
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": "stage71-fail-closed-track-pair-gate-v2",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "upstream_report": str(upstream_path.resolve()),
        "upstream_report_sha256": sha256(upstream_path),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)

    cache: dict[tuple[str, str, str, float], dict[str, Any]] = {}

    def evaluate(
        dataset: str,
        manifest: Path,
        model_name: str,
        checkpoint: Path,
        checkpoint_sha: str,
        head: str,
        threshold: float,
    ) -> dict[str, Any]:
        require_sha(checkpoint, checkpoint_sha, f"{model_name} checkpoint")
        key = (dataset, checkpoint_sha.lower(), head, threshold)
        if key in cache:
            return cache[key]
        slug = safe_name(f"{dataset}-{model_name}-{head}-{threshold:.4f}")
        output = args.output_root / f"{slug}.json"
        log = args.output_root / f"{slug}.log"
        state["active"] = {
            "dataset": dataset,
            "model": model_name,
            "head": head,
            "threshold": threshold,
        }
        state["updated_at"] = now()
        atomic_json(args.state, state)
        run(
            [
                args.python,
                str(pins["v2_sweeper"][0]),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--split",
                "validation",
                "--minimum-window-frames",
                "3",
                "--thresholds",
                str(threshold),
                "--windows",
                "3",
                "5",
                "--minimum-shares",
                "0.50",
                "0.60",
                "0.70",
                "--minimum-margins",
                "0.05",
                "0.10",
                "0.15",
                "--batch-size",
                "128",
                "--workers",
                "4",
                "--device",
                args.device,
            ],
            log,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("schema_version") != "attribute-track-fusion-sweep-fail-closed-v2":
            raise RuntimeError("v2 sweeper did not emit fail-closed evidence")
        result = compact_selected(report, head)
        result["report"] = str(output.resolve())
        result["report_sha256"] = sha256(output)
        result["checkpoint_sha256"] = checkpoint_sha.lower()
        cache[key] = result
        return result

    production_sha = args.expected_production_checkpoint_sha256
    production_body_vfg = evaluate(
        "vfg", args.vfg_manifest, "production", args.production_checkpoint,
        production_sha, "body_type", production_body_threshold,
    )
    production_color_vfg = evaluate(
        "vfg", args.vfg_manifest, "production", args.production_checkpoint,
        production_sha, "color", production_color_threshold,
    )
    production_body_ua = evaluate(
        "ua", args.ua_manifest, "production", args.production_checkpoint,
        production_sha, "body_type", production_body_threshold,
    )

    pairs = copy.deepcopy(upstream["pairs"])
    pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
    for pair_id in passing_ids:
        pair = pairs_by_id[pair_id]
        body_checkpoint = Path(pair["body_checkpoint"])
        color_checkpoint = Path(pair["color_checkpoint"])
        body_threshold = number(
            pair["vfg7"]["body"]["selected_threshold"]["threshold"]
        )
        color_threshold = number(
            pair["vfg7"]["color"]["selected_threshold"]["threshold"]
        )
        body_vfg = evaluate(
            "vfg", args.vfg_manifest, pair["body_model"], body_checkpoint,
            pair["body_checkpoint_sha256"], "body_type", body_threshold,
        )
        color_vfg = evaluate(
            "vfg", args.vfg_manifest, pair["color_model"], color_checkpoint,
            pair["color_checkpoint_sha256"], "color", color_threshold,
        )
        body_ua = evaluate(
            "ua", args.ua_manifest, pair["body_model"], body_checkpoint,
            pair["body_checkpoint_sha256"], "body_type", body_threshold,
        )
        gates, comparison = pair_gates(
            body_vfg,
            color_vfg,
            body_ua,
            production_body_vfg,
            production_color_vfg,
        )
        pair["fail_closed_v2"] = {
            "body_vfg": body_vfg,
            "color_vfg": color_vfg,
            "body_ua": body_ua,
            "gates": gates,
            "comparison_to_production": comparison,
        }
        pair["gates"]["fail_closed_v2_track_gate"] = all(gates.values())
        pair["screen_status"] = (
            "pass" if all(pair["gates"].values()) else "fail_closed"
        )

    narrowed_passing = [
        pair_id
        for pair_id in passing_ids
        if pairs_by_id[pair_id]["screen_status"] == "pass"
    ]
    report = {
        "schema_version": "stage71-specialist-student-pair-validation-fail-closed-v2",
        "created_at": now(),
        "status": (
            "pass_pairs_available" if narrowed_passing
            else "complete_all_pairs_rejected_by_fail_closed_v2"
        ),
        "upstream_report": str(upstream_path.resolve()),
        "upstream_report_sha256": sha256(upstream_path),
        "body_models": upstream.get("body_models", []),
        "color_models": upstream.get("color_models", []),
        "pairs": pairs,
        "passing_pairs": narrowed_passing,
        "production_fail_closed_v2": {
            "body_vfg": production_body_vfg,
            "color_vfg": production_color_vfg,
            "body_ua": production_body_ua,
        },
        "policy": {
            "validation_only": True,
            "fail_closed_v2_required_before_test": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
        "decision": (
            "only v2-passing pairs may proceed to the one-time independent test"
            if narrowed_passing
            else "reject every pair before test, backend, frozen replay, or deployment"
        ),
    }
    report_path = args.output_root / "pair-validation-fail-closed-v2-report.json"
    atomic_json(report_path, report)
    state.update(
        {
            "status": "complete",
            "updated_at": now(),
            "active": None,
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "passing_pairs": narrowed_passing,
        }
    )
    atomic_json(args.state, state)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passing_pairs": narrowed_passing,
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
