#!/usr/bin/env python3
"""Run the Stage77 validation-only body gate after Stage76 training finishes.

The runner waits on the isolated Stage76 state, then performs only validation
work: fine static threshold selection on VFG-7, fixed-threshold stratified
evaluation on VFG-7 and the secondary hard set, and fail-closed track fusion.
It never accepts a test split and has no frozen-video or deployment input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_COMPLETE = "complete_validation_only_two_phase"
PRECISION_GATE = 0.93
COVERAGE_GATE = 0.45
COMPLEX_COVERAGE_GAIN_GATE = 0.15
STABILITY_GATE = 0.95


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def threshold_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = report.get("threshold_sweep", {}).get("body_type", {})
    candidates: list[dict[str, Any]] = []
    for threshold_text, metrics in sweep.items():
        if not isinstance(metrics, dict):
            continue
        candidates.append(
            {
                "threshold": float(threshold_text),
                "precision": float(metrics.get("high_confidence_precision", 0.0)),
                "coverage": float(metrics.get("high_confidence_coverage", 0.0)),
                "selected": int(metrics.get("high_confidence_selected", 0)),
                "evaluated": int(metrics.get("evaluated", 0)),
            }
        )
    if not candidates:
        raise ValueError("body threshold sweep is empty")
    return candidates


def select_body_threshold(
    report: dict[str, Any], minimum_precision: float = PRECISION_GATE
) -> dict[str, Any]:
    candidates = threshold_candidates(report)
    eligible = [
        item
        for item in candidates
        if item["selected"] > 0 and item["precision"] >= minimum_precision
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["coverage"],
                item["precision"],
                -item["threshold"],
            ),
        )
        return {**selected, "precision_gate_found": True}
    fallback = max(
        candidates,
        key=lambda item: (
            item["precision"],
            item["coverage"],
            -item["threshold"],
        ),
    )
    return {**fallback, "precision_gate_found": False}


def fixed_head_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("body_type", {})
    return {
        "evaluated": int(metrics.get("evaluated", 0)),
        "selected": int(metrics.get("high_confidence_selected", 0)),
        "precision": float(metrics.get("high_confidence_precision", 0.0)),
        "coverage": float(metrics.get("high_confidence_coverage", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
    }


def selected_track_metrics(report: dict[str, Any]) -> dict[str, Any]:
    head = report.get("body_type", {})
    selected = head.get("selected", {})
    final_window = selected.get("final_window", {})
    single = selected.get("single", {})
    stability = selected.get("stability", {})
    return {
        "selection_status": head.get("selection_status", "missing"),
        "threshold": float(selected.get("threshold", 0.0)),
        "window": int(selected.get("window", 0)),
        "minimum_share": float(selected.get("minimum_share", 0.0)),
        "minimum_margin": float(selected.get("minimum_margin", 0.0)),
        "single_precision": float(single.get("precision") or 0.0),
        "single_coverage": float(single.get("coverage") or 0.0),
        "final_precision": float(final_window.get("precision") or 0.0),
        "final_coverage": float(final_window.get("coverage") or 0.0),
        "effective_unknown_rate": float(
            final_window.get("effective_unknown_rate") or 0.0
        ),
        "weighted_stability_rate": float(
            stability.get("weighted_stability_rate") or 0.0
        ),
        "label_switches_total": int(stability.get("label_switches_total") or 0),
    }


def body_gate_decision(
    candidate_static: dict[str, Any],
    production_static: dict[str, Any],
    candidate_hard: dict[str, Any],
    candidate_track: dict[str, Any],
) -> dict[str, Any]:
    gain = candidate_static["coverage"] - production_static["coverage"]
    gates = {
        "static_precision": candidate_static["precision"] >= PRECISION_GATE,
        "static_coverage": candidate_static["coverage"] >= COVERAGE_GATE,
        "complex_coverage_gain": gain >= COMPLEX_COVERAGE_GAIN_GATE,
        "secondary_hard_precision": candidate_hard["precision"] >= PRECISION_GATE,
        "secondary_hard_coverage": candidate_hard["coverage"] >= COVERAGE_GATE,
        "track_precision": candidate_track["final_precision"] >= PRECISION_GATE,
        "track_coverage": candidate_track["final_coverage"] >= COVERAGE_GATE,
        "track_stability": (
            candidate_track["weighted_stability_rate"] >= STABILITY_GATE
        ),
    }
    return {
        "status": (
            "body_component_validation_qualified_pending_color_pair_and_final_test"
            if all(gates.values())
            else "body_component_rejected_validation_gate_failure"
        ),
        "coverage_gain_vs_production": gain,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def run_command(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; log={log}"
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--training-state", type=Path, required=True)
    value.add_argument("--candidate-checkpoint", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--state", type=Path, required=True)
    value.add_argument("--evaluator", type=Path, required=True)
    value.add_argument("--track-sweeper", type=Path, required=True)
    value.add_argument("--labels", type=Path, required=True)
    value.add_argument("--vfg-manifest", type=Path, required=True)
    value.add_argument("--hard-manifest", type=Path, required=True)
    value.add_argument("--production-checkpoint", type=Path, required=True)
    value.add_argument("--stage71-checkpoint", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=21600)
    return value


def assert_validation_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.training_state,
        args.evaluator,
        args.track_sweeper,
        args.labels,
        args.vfg_manifest,
        args.hard_manifest,
        args.production_checkpoint,
        args.stage71_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage77 evidence")
    for manifest in (args.vfg_manifest, args.hard_manifest):
        lowered = str(manifest).lower()
        if "test" in lowered or "60s" in lowered or "36-48" in lowered:
            raise ValueError(f"non-validation input rejected: {manifest}")


def wait_for_training(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        training = read_json(args.training_state)
        status = str(training.get("status", "missing"))
        if status == TRAINING_COMPLETE:
            return training
        if status != "running":
            raise RuntimeError(f"Stage76 did not complete successfully: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage76 wait timed out")
        time.sleep(max(5, args.poll_seconds))


def validate_candidate(training: dict[str, Any], checkpoint: Path) -> str:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records = training.get("completed", [])
    matches = [
        record
        for record in records
        if record.get("phase") == "cctv_finetune"
        and Path(str(record.get("best_checkpoint", ""))) == checkpoint
    ]
    if len(matches) != 1:
        raise RuntimeError("Stage76 state does not authorize the candidate checkpoint")
    actual = sha256(checkpoint)
    expected = str(matches[0].get("best_checkpoint_sha256", "")).lower()
    if expected and actual != expected:
        raise RuntimeError("Stage76 candidate checkpoint SHA256 mismatch")
    return actual


def main() -> int:
    args = parser().parse_args()
    state: dict[str, Any] = {
        "schema_version": "stage77-body-validation-after-stage76-v1",
        "status": "waiting_for_stage76",
        "created_at": now(),
        "updated_at": now(),
        "split": "validation",
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    try:
        assert_validation_inputs(args)
        atomic_json(args.state, state)
        training = wait_for_training(args)
        candidate_sha = validate_candidate(training, args.candidate_checkpoint)
        args.output_root.mkdir(parents=True, exist_ok=False)
        state.update(
            {
                "status": "running_validation",
                "updated_at": now(),
                "candidate_checkpoint": str(args.candidate_checkpoint),
                "candidate_checkpoint_sha256": candidate_sha,
            }
        )
        atomic_json(args.state, state)

        fine_thresholds = [f"{value / 1000:.3f}" for value in range(500, 991)]
        static_sweep_reports: dict[str, Path] = {}
        for name, checkpoint in (
            ("production", args.production_checkpoint),
            ("stage76", args.candidate_checkpoint),
        ):
            report = args.output_root / f"vfg7-static-sweep-{name}.json"
            run_command(
                [
                    str(Path(os.sys.executable)),
                    str(args.evaluator),
                    "--manifest",
                    str(args.vfg_manifest),
                    "--checkpoint",
                    str(checkpoint),
                    "--labels",
                    str(args.labels),
                    "--output",
                    str(report),
                    "--split",
                    "validation",
                    "--batch-size",
                    "128",
                    "--workers",
                    "4",
                    "--device",
                    args.device,
                    "--type-threshold",
                    "0.75",
                    "--color-threshold",
                    "0.99",
                    "--threshold-sweep",
                    *fine_thresholds,
                ],
                args.output_root / f"vfg7-static-sweep-{name}.log",
            )
            static_sweep_reports[name] = report

        production_selection = select_body_threshold(
            read_json(static_sweep_reports["production"])
        )
        candidate_selection = select_body_threshold(
            read_json(static_sweep_reports["stage76"])
        )

        fixed_reports: dict[str, Path] = {}
        for dataset_name, manifest in (
            ("vfg7", args.vfg_manifest),
            ("hard", args.hard_manifest),
        ):
            for name, checkpoint, selection in (
                ("production", args.production_checkpoint, production_selection),
                ("stage76", args.candidate_checkpoint, candidate_selection),
            ):
                report = args.output_root / f"{dataset_name}-static-fixed-{name}.json"
                run_command(
                    [
                        str(Path(os.sys.executable)),
                        str(args.evaluator),
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--labels",
                        str(args.labels),
                        "--output",
                        str(report),
                        "--split",
                        "validation",
                        "--batch-size",
                        "128",
                        "--workers",
                        "4",
                        "--device",
                        args.device,
                        "--type-threshold",
                        f"{selection['threshold']:.3f}",
                        "--color-threshold",
                        "0.99",
                    ],
                    args.output_root / f"{dataset_name}-static-fixed-{name}.log",
                )
                fixed_reports[f"{dataset_name}_{name}"] = report

        track_reports: dict[str, Path] = {}
        track_thresholds = [f"{value / 100:.2f}" for value in range(50, 100)]
        for name, checkpoint in (
            ("production", args.production_checkpoint),
            ("stage71", args.stage71_checkpoint),
            ("stage76", args.candidate_checkpoint),
        ):
            report = args.output_root / f"vfg7-track-fusion-{name}.json"
            run_command(
                [
                    str(Path(os.sys.executable)),
                    str(args.track_sweeper),
                    "--manifest",
                    str(args.vfg_manifest),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(report),
                    "--split",
                    "validation",
                    "--minimum-window-frames",
                    "3",
                    "--thresholds",
                    *track_thresholds,
                    "--windows",
                    "3",
                    "5",
                    "--minimum-shares",
                    "0.50",
                    "0.60",
                    "0.70",
                    "0.80",
                    "0.90",
                    "0.95",
                    "--minimum-margins",
                    "0.00",
                    "0.10",
                    "0.20",
                    "0.30",
                    "0.40",
                    "0.50",
                    "--batch-size",
                    "128",
                    "--workers",
                    "4",
                    "--device",
                    args.device,
                ],
                args.output_root / f"vfg7-track-fusion-{name}.log",
            )
            track_reports[name] = report

        production_static = fixed_head_metrics(
            read_json(fixed_reports["vfg7_production"])
        )
        candidate_static = fixed_head_metrics(
            read_json(fixed_reports["vfg7_stage76"])
        )
        candidate_hard = fixed_head_metrics(
            read_json(fixed_reports["hard_stage76"])
        )
        production_track = selected_track_metrics(
            read_json(track_reports["production"])
        )
        stage71_track = selected_track_metrics(read_json(track_reports["stage71"]))
        candidate_track = selected_track_metrics(read_json(track_reports["stage76"]))
        decision = body_gate_decision(
            candidate_static,
            production_static,
            candidate_hard,
            candidate_track,
        )
        evidence_paths = [
            *static_sweep_reports.values(),
            *fixed_reports.values(),
            *track_reports.values(),
        ]
        evidence = {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in evidence_paths
        }
        report = {
            "schema_version": "stage77-body-validation-decision-v1",
            "created_at": now(),
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "candidate_checkpoint_sha256": candidate_sha,
            "threshold_selection": {
                "source": "VFG-7 validation only",
                "production": production_selection,
                "stage76": candidate_selection,
            },
            "vfg7_static": {
                "production": production_static,
                "stage76": candidate_static,
            },
            "secondary_hard_static": {
                "production": fixed_head_metrics(
                    read_json(fixed_reports["hard_production"])
                ),
                "stage76": candidate_hard,
            },
            "vfg7_track": {
                "production": production_track,
                "stage71": stage71_track,
                "stage76": candidate_track,
            },
            "decision": decision,
            "evidence": evidence,
            "policy": {
                "split": "validation",
                "test_accessed": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
                "deployment_paused_by_user": True,
            },
        }
        decision_path = args.output_root / "stage77-body-validation-decision-v1.json"
        atomic_json(decision_path, report)
        (decision_path.with_suffix(decision_path.suffix + ".sha256")).write_text(
            f"{sha256(decision_path)}  {decision_path.name}\n", encoding="utf-8"
        )
        state.update(
            {
                "status": "complete_validation_only",
                "updated_at": now(),
                "decision": decision,
                "decision_report": str(decision_path),
                "decision_report_sha256": sha256(decision_path),
            }
        )
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        state.update(
            {
                "status": "failed_closed",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if not args.state.exists() or read_json(args.state).get("status") != "complete_validation_only":
            atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
