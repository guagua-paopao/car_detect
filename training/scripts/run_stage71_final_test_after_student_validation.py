#!/usr/bin/env python3
"""Run one final test for one validation-selected Stage71 student pair.

This script is deliberately fail-closed:

* it waits for the validation-only pair screen to complete;
* it selects exactly one pair using validation evidence only;
* it evaluates that pair and the frozen production baseline on test once;
* it never reads the frozen 60-second acceptance video;
* it never exports, deploys, or mutates the production model.

Backend and frozen-video gates are separate later stages and may only consume a
``pass_backend_eligible`` report produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


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
            f"final-test command failed ({result.returncode}); see {log}"
        )


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_validation(
    state_path: Path,
    session: str,
    timeout_hours: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = state.get("status")
            if status == "complete":
                return state
            if status not in {"waiting_for_training", "running"}:
                raise RuntimeError(
                    f"student validation failed closed with status={status!r}"
                )
        if not tmux_alive(session):
            raise RuntimeError(
                "student-validation session ended without complete evidence"
            )
        time.sleep(30)
    raise TimeoutError("timed out waiting for Stage71 student validation")


def validate_validation_report(
    validation_state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    report_path = Path(str(validation_state.get("report", "")))
    expected_sha = str(validation_state.get("report_sha256", ""))
    if not expected_sha:
        raise RuntimeError("validation state has no report SHA256")
    require_sha(report_path, expected_sha, "student validation report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = report.get("policy", {})
    if policy.get("validation_only") is not True:
        raise RuntimeError("student pair screen is not validation-only")
    for key in (
        "test_accessed",
        "frozen_video_used",
        "production_model_modified",
        "backend_gates_run",
        "deployment_performed",
    ):
        if policy.get(key) is not False:
            raise RuntimeError(f"validation policy violation: {key}")
    passing = set(report.get("passing_pairs", []))
    if not passing:
        raise RuntimeError("no validation-passing student pair; test remains locked")
    candidates = [
        pair
        for pair in report.get("pairs", [])
        if pair.get("pair_id") in passing and pair.get("screen_status") == "pass"
    ]
    if len(candidates) != len(passing):
        raise RuntimeError("passing-pair list does not match pair evidence")
    return report_path, report


def number(value: Any, default: float = 0.0) -> float:
    return float(default if value is None else value)


def validation_pair_key(pair: dict[str, Any]) -> tuple[Any, ...]:
    """Deterministically rank pairs without consulting any test evidence."""

    hard = pair["hard"]
    body = hard["body_type"]
    color = hard["color"]
    vfg = pair.get("vfg7", {})
    body_track = vfg.get("body", {}).get("track", {})
    color_track = vfg.get("color", {}).get("track", {})
    margins = (
        number(body.get("precision")) - 0.93,
        number(body.get("coverage")) - 0.45,
        number(color.get("precision")) - 0.93,
        number(color.get("coverage")) - 0.25,
    )
    return (
        min(margins),
        number(body.get("coverage")) + number(color.get("coverage")),
        number(body_track.get("body_fusion5_final", {}).get("coverage"))
        + number(color_track.get("color_fusion5_final", {}).get("coverage")),
        number(body.get("precision")) + number(color.get("precision")),
        # Stable final tie-breaker independent of filesystem ordering.
        str(pair.get("pair_id", "")),
    )


def selected_pair(report: dict[str, Any]) -> dict[str, Any]:
    passing = set(report["passing_pairs"])
    pairs = [pair for pair in report["pairs"] if pair.get("pair_id") in passing]
    return max(pairs, key=validation_pair_key)


def static_metrics(report: dict[str, Any], head: str) -> dict[str, Any]:
    source = report[head]
    return {
        "evaluated": int(source["evaluated"]),
        "precision": number(source["high_confidence_precision"]),
        "coverage": number(source["high_confidence_coverage"]),
        "selected": int(source["high_confidence_selected"]),
        "accuracy": number(source["accuracy"]),
        "macro_f1": number(source["macro_f1"]),
        "predicted_unknown_rate": number(source["predicted_unknown_rate"]),
    }


def fused_metrics(report: dict[str, Any], head: str) -> dict[str, Any]:
    selected = report[head].get("selected")
    if not selected:
        return {"selection_status": "no_selection"}
    final = selected.get("final_window", {})
    stability = selected.get("stability", {})
    return {
        "threshold": selected.get("threshold"),
        "window": selected.get("window"),
        "minimum_share": selected.get("minimum_share"),
        "minimum_margin": selected.get("minimum_margin"),
        "evaluated": final.get("evaluated"),
        "precision": final.get("precision"),
        "coverage": final.get("coverage"),
        "effective_unknown_rate": final.get("effective_unknown_rate"),
        "weighted_stability_rate": stability.get("weighted_stability_rate"),
        "label_switches_total": stability.get("label_switches_total"),
        "abstention_rate": stability.get("abstention_rate"),
    }


def relative_unknown_reduction(baseline_coverage: float, candidate_coverage: float) -> float:
    baseline_unknown = 1.0 - baseline_coverage
    candidate_unknown = 1.0 - candidate_coverage
    return (
        (baseline_unknown - candidate_unknown) / baseline_unknown
        if baseline_unknown > 0.0
        else 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-state", type=Path, required=True)
    parser.add_argument("--validation-session", required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--expected-hard-manifest-sha256", required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--expected-vfg-manifest-sha256", required=True)
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-production-checkpoint-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--expected-static-evaluator-sha256", required=True)
    parser.add_argument("--expected-track-evaluator-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--production-type-threshold", type=float, default=0.75)
    parser.add_argument("--production-color-threshold", type=float, default=0.70)
    args = parser.parse_args()

    if args.output_root.exists() or args.state.exists():
        raise RuntimeError("refusing to overwrite final-test evidence")
    pins = {
        "hard manifest": (args.hard_manifest, args.expected_hard_manifest_sha256),
        "UA manifest": (args.ua_manifest, args.expected_ua_manifest_sha256),
        "VFG manifest": (args.vfg_manifest, args.expected_vfg_manifest_sha256),
        "production checkpoint": (
            args.production_checkpoint,
            args.expected_production_checkpoint_sha256,
        ),
        "labels": (args.labels, args.expected_labels_sha256),
        "static evaluator": (
            args.scripts_root / "evaluate_attribute_baseline.py",
            args.expected_static_evaluator_sha256,
        ),
        "track evaluator": (
            args.scripts_root / "evaluate_attribute_track_fusion.py",
            args.expected_track_evaluator_sha256,
        ),
    }
    for label, (path, expected) in pins.items():
        require_sha(path, expected, label)

    args.output_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "stage71-final-test-v1",
        "status": "waiting_for_student_validation",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "test_accessed": False,
        "test_used_for_selection": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)

    try:
        validation_state = wait_for_validation(
            args.validation_state,
            args.validation_session,
            args.timeout_hours,
        )
        validation_report_path, validation_report = validate_validation_report(
            validation_state
        )
        pair = selected_pair(validation_report)
        body_checkpoint = Path(pair["body_checkpoint"])
        color_checkpoint = Path(pair["color_checkpoint"])
        require_sha(body_checkpoint, pair["body_checkpoint_sha256"], "body checkpoint")
        require_sha(color_checkpoint, pair["color_checkpoint_sha256"], "color checkpoint")

        body_threshold = number(pair["hard"]["body_type"]["threshold"])
        color_threshold = number(pair["hard"]["color"]["threshold"])
        if not 0.0 < body_threshold <= 1.0 or not 0.0 < color_threshold <= 1.0:
            raise RuntimeError("invalid validation-selected threshold")

        state.update(
            {
                "status": "running_single_selected_pair_test",
                "updated_at": utc_now(),
                "test_accessed": True,
                "validation_state": str(args.validation_state.resolve()),
                "validation_state_sha256": sha256(args.validation_state),
                "validation_report": str(validation_report_path.resolve()),
                "validation_report_sha256": sha256(validation_report_path),
                "selection_policy": "one pair selected only from validation gates and deterministic validation ranking",
                "selected_pair": pair["pair_id"],
                "body_checkpoint": str(body_checkpoint.resolve()),
                "body_checkpoint_sha256": sha256(body_checkpoint),
                "color_checkpoint": str(color_checkpoint.resolve()),
                "color_checkpoint_sha256": sha256(color_checkpoint),
                "thresholds": {
                    "candidate_body": body_threshold,
                    "candidate_color": color_threshold,
                    "production_body": args.production_type_threshold,
                    "production_color": args.production_color_threshold,
                },
            }
        )
        atomic_json(args.state, state)

        static_evaluator = args.scripts_root / "evaluate_attribute_baseline.py"
        track_evaluator = args.scripts_root / "evaluate_attribute_track_fusion.py"
        models = {
            "production": (
                args.production_checkpoint,
                args.production_type_threshold,
                args.production_color_threshold,
            ),
            "body": (body_checkpoint, body_threshold, color_threshold),
            "color": (color_checkpoint, body_threshold, color_threshold),
        }
        static_reports: dict[str, dict[str, dict[str, Any]]] = {}
        for dataset, manifest in (("hard", args.hard_manifest), ("vfg", args.vfg_manifest)):
            static_reports[dataset] = {}
            for model_name, (checkpoint, type_threshold, model_color_threshold) in models.items():
                output = args.output_root / f"{dataset}-{safe_name(model_name)}-test.json"
                run(
                    [
                        args.python,
                        str(static_evaluator),
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--labels",
                        str(args.labels),
                        "--output",
                        str(output),
                        "--split",
                        "test",
                        "--batch-size",
                        "128",
                        "--workers",
                        "4",
                        "--device",
                        args.device,
                        "--type-threshold",
                        str(type_threshold),
                        "--color-threshold",
                        str(model_color_threshold),
                    ],
                    args.output_root / f"{dataset}-{safe_name(model_name)}-test.log",
                )
                static_reports[dataset][model_name] = json.loads(
                    output.read_text(encoding="utf-8")
                )

        track_reports: dict[str, dict[str, dict[str, Any]]] = {}
        for dataset, manifest in (("ua", args.ua_manifest), ("vfg", args.vfg_manifest)):
            track_reports[dataset] = {}
            for model_name, (checkpoint, type_threshold, model_color_threshold) in models.items():
                output = args.output_root / f"{dataset}-{safe_name(model_name)}-track-test.json"
                run(
                    [
                        args.python,
                        str(track_evaluator),
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                        "--split",
                        "test",
                        "--type-threshold",
                        str(type_threshold),
                        "--color-threshold",
                        str(model_color_threshold),
                        "--fusion-windows",
                        "3",
                        "5",
                        "--batch-size",
                        "128",
                        "--workers",
                        "4",
                        "--device",
                        args.device,
                    ],
                    args.output_root / f"{dataset}-{safe_name(model_name)}-track-test.log",
                )
                track_reports[dataset][model_name] = json.loads(
                    output.read_text(encoding="utf-8")
                )

        hard_production_body = static_metrics(static_reports["hard"]["production"], "body_type")
        hard_production_color = static_metrics(static_reports["hard"]["production"], "color")
        hard_body = static_metrics(static_reports["hard"]["body"], "body_type")
        hard_color = static_metrics(static_reports["hard"]["color"], "color")
        body_gain = hard_body["coverage"] - hard_production_body["coverage"]
        color_unknown_reduction = relative_unknown_reduction(
            hard_production_color["coverage"], hard_color["coverage"]
        )

        vfg_body = static_metrics(static_reports["vfg"]["body"], "body_type")
        vfg_color = static_metrics(static_reports["vfg"]["color"], "color")
        ua_body = fused_metrics(track_reports["ua"]["body"], "body_type")
        vfg_body_track = fused_metrics(track_reports["vfg"]["body"], "body_type")
        vfg_color_track = fused_metrics(track_reports["vfg"]["color"], "color")

        # A long-running test must not silently accept inputs or checkpoints
        # that changed after the initial preflight.
        for label, (path, expected) in pins.items():
            require_sha(path, expected, label)
        require_sha(body_checkpoint, pair["body_checkpoint_sha256"], "body checkpoint")
        require_sha(color_checkpoint, pair["color_checkpoint_sha256"], "color checkpoint")

        gates = {
            "hard_body_precision_gte_0_93": hard_body["precision"] >= 0.93,
            "hard_body_coverage_gte_0_45": hard_body["coverage"] >= 0.45,
            "hard_color_precision_gte_0_93": hard_color["precision"] >= 0.93,
            "hard_color_coverage_gte_0_25": hard_color["coverage"] >= 0.25,
            "hard_body_coverage_gain_gte_15pp": body_gain >= 0.15,
            "hard_color_unknown_reduction_gte_20pct": color_unknown_reduction >= 0.20,
            "ua_body_precision_gte_0_93": number(ua_body.get("precision")) >= 0.93,
            "ua_body_coverage_gte_0_45": number(ua_body.get("coverage")) >= 0.45,
            "ua_body_stability_gte_0_95": number(ua_body.get("weighted_stability_rate")) >= 0.95,
            "vfg_static_body_precision_gte_0_93": vfg_body["precision"] >= 0.93,
            "vfg_static_body_coverage_gte_0_45": vfg_body["coverage"] >= 0.45,
            "vfg_static_color_precision_gte_0_93": vfg_color["precision"] >= 0.93,
            "vfg_static_color_coverage_gte_0_25": vfg_color["coverage"] >= 0.25,
            "vfg_track_body_precision_gte_0_93": number(vfg_body_track.get("precision")) >= 0.93,
            "vfg_track_body_coverage_gte_0_45": number(vfg_body_track.get("coverage")) >= 0.45,
            "vfg_track_color_precision_gte_0_93": number(vfg_color_track.get("precision")) >= 0.93,
            "vfg_track_color_coverage_gte_0_25": number(vfg_color_track.get("coverage")) >= 0.25,
            "vfg_track_body_stability_gte_0_95": number(vfg_body_track.get("weighted_stability_rate")) >= 0.95,
            "vfg_track_color_stability_gte_0_95": number(vfg_color_track.get("weighted_stability_rate")) >= 0.95,
        }
        passed = all(gates.values())
        final_report = {
            "schema_version": "stage71-final-test-report-v1",
            "created_at": utc_now(),
            "status": "pass_backend_eligible" if passed else "fail_closed_before_backend",
            "selection": {
                "source": "validation_only",
                "pair_id": pair["pair_id"],
                "validation_report": str(validation_report_path.resolve()),
                "validation_report_sha256": sha256(validation_report_path),
                "test_used_for_selection": False,
            },
            "candidate": {
                "body_checkpoint": str(body_checkpoint.resolve()),
                "body_checkpoint_sha256": sha256(body_checkpoint),
                "color_checkpoint": str(color_checkpoint.resolve()),
                "color_checkpoint_sha256": sha256(color_checkpoint),
                "thresholds": {
                    "body_type": body_threshold,
                    "color": color_threshold,
                },
            },
            "hard_test": {
                "production": {
                    "body_type": hard_production_body,
                    "color": hard_production_color,
                },
                "candidate": {"body_type": hard_body, "color": hard_color},
                "comparison": {
                    "body_coverage_gain_percentage_points": body_gain * 100.0,
                    "color_unknown_relative_reduction": color_unknown_reduction,
                },
            },
            "ua_track_test": {"candidate_body_type": ua_body},
            "vfg_test": {
                "static": {"body_type": vfg_body, "color": vfg_color},
                "track": {
                    "body_type": vfg_body_track,
                    "color": vfg_color_track,
                },
            },
            "gates": gates,
            "evidence": {
                "static_reports": {
                    dataset: {
                        name: {
                            "path": str((args.output_root / f"{dataset}-{safe_name(name)}-test.json").resolve()),
                            "sha256": sha256(args.output_root / f"{dataset}-{safe_name(name)}-test.json"),
                        }
                        for name in models
                    }
                    for dataset in ("hard", "vfg")
                },
                "track_reports": {
                    dataset: {
                        name: {
                            "path": str((args.output_root / f"{dataset}-{safe_name(name)}-track-test.json").resolve()),
                            "sha256": sha256(args.output_root / f"{dataset}-{safe_name(name)}-track-test.json"),
                        }
                        for name in models
                    }
                    for dataset in ("ua", "vfg")
                },
            },
            "policy": {
                "test_accessed": True,
                "test_used_for_selection": False,
                "one_validation_selected_pair_tested": True,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
                "deployment_paused_by_user": True,
            },
            "next_action": (
                "export isolated candidate ONNX and run ONNX/TensorRT/C++/real-engine gates"
                if passed
                else "reject Stage71 student pair and plan the next data/model iteration"
            ),
        }
        report_path = args.output_root / "final-test-report.json"
        atomic_json(report_path, final_report)
        state.update(
            {
                "status": final_report["status"],
                "updated_at": utc_now(),
                "report": str(report_path.resolve()),
                "report_sha256": sha256(report_path),
                "gates": gates,
                "backend_eligible": passed,
            }
        )
        atomic_json(args.state, state)
        print(
            json.dumps(
                {
                    "status": final_report["status"],
                    "selected_pair": pair["pair_id"],
                    "gates": gates,
                    "report": str(report_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0 if passed else 2
    except Exception as exc:
        state.update(
            {
                "status": "failed_closed",
                "updated_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
