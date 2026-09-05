#!/usr/bin/env python3
"""Wait for specialist training, then screen body/color pairs on validation only."""

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


THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86,
    0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995,
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"validation command failed ({result.returncode}); see {log}")


def wait_for_training(path: Path, session: str, timeout_hours: float) -> dict:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "complete_validation_only":
                return state
            if state.get("status") != "running":
                raise RuntimeError(f"unexpected training state: {state.get('status')}")
        alive = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
        if not alive:
            raise RuntimeError("training session ended without complete evidence")
        time.sleep(30)
    raise TimeoutError("timed out waiting for specialist training")


def select_threshold(sweep: dict, precision_gate: float = 0.93) -> dict:
    rows = [{
        "threshold": float(threshold),
        "precision": float(metrics["high_confidence_precision"]),
        "coverage": float(metrics["high_confidence_coverage"]),
        "selected": int(metrics["high_confidence_selected"]),
    } for threshold, metrics in sweep.items()]
    passing = [row for row in rows if row["precision"] >= precision_gate and row["selected"] > 0]
    chosen = (
        max(passing, key=lambda row: (row["coverage"], row["precision"], -row["threshold"]))
        if passing else max(rows, key=lambda row: (row["precision"], row["coverage"], row["threshold"]))
    )
    return {**chosen, "precision_gate_pass": bool(passing)}


def compact_track(selected: dict | None) -> dict:
    if not selected:
        return {"selection_status": "no_selection"}
    final = selected.get("final_window", {})
    stability = selected.get("stability", {})
    return {
        "threshold": selected.get("threshold"), "window": selected.get("window"),
        "minimum_share": selected.get("minimum_share"), "minimum_margin": selected.get("minimum_margin"),
        "evaluated": final.get("evaluated"), "precision": final.get("precision"),
        "coverage": final.get("coverage"), "effective_unknown_rate": final.get("effective_unknown_rate"),
        "weighted_stability_rate": stability.get("weighted_stability_rate"),
        "label_switches_total": stability.get("label_switches_total"),
        "abstention_rate": stability.get("abstention_rate"),
    }


def validate_training_contract(training: dict, expected_matrix_sha256: str, expected_candidates: int) -> None:
    if training.get("matrix_sha256", "").lower() != expected_matrix_sha256.lower():
        raise RuntimeError("training state is not bound to the pinned training matrix")
    if training.get("failed") or len(training.get("completed", [])) != expected_candidates:
        raise RuntimeError("specialist training is incomplete or contains a failed candidate")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if training.get(key) is not False:
            raise RuntimeError(f"specialist training policy violation: {key}")


def vfg_pair_gates(body: dict, color: dict, production: dict) -> tuple[dict, dict]:
    body_threshold = body["selected_thresholds"]["body_type"]
    color_threshold = color["selected_thresholds"]["color"]
    body_track = body["track_validation"]["body_fusion5_final"]
    color_track = color["track_validation"]["color_fusion5_final"]
    body_stability = body["track_validation"]["body_fusion5_stability"].get("weighted_stability_rate") or 0.0
    color_stability = color["track_validation"]["color_fusion5_stability"].get("weighted_stability_rate") or 0.0
    baseline_body = production["track_validation"]["body_fusion5_final"].get("coverage") or 0.0
    baseline_unknown = production["track_validation"]["color_fusion5_stability"].get("abstention_rate") or 0.0
    body_coverage = body_track.get("coverage") or 0.0
    color_unknown = color["track_validation"]["color_fusion5_stability"].get("abstention_rate") or 0.0
    body_gain = body_coverage - baseline_body
    unknown_reduction = ((baseline_unknown - color_unknown) / baseline_unknown) if baseline_unknown > 0 else 0.0
    gates = {
        "static_body": body_threshold["precision_gate_pass"] and body_threshold["coverage"] >= 0.45,
        "static_color": color_threshold["precision_gate_pass"] and color_threshold["coverage"] >= 0.25,
        "track_body": (body_track.get("precision") or 0.0) >= 0.93 and body_coverage >= 0.45,
        "track_color": (color_track.get("precision") or 0.0) >= 0.93 and (color_track.get("coverage") or 0.0) >= 0.25,
        "track_stability": body_stability >= 0.95 and color_stability >= 0.95,
        "type_coverage_gain_15pp": body_gain >= 0.15,
        "color_unknown_reduction_20pct": unknown_reduction >= 0.20,
    }
    return gates, {
        "body_fusion5_final_coverage_gain_percentage_points": body_gain * 100.0,
        "color_fusion5_unknown_relative_reduction": unknown_reduction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--expected-training-matrix-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-session", required=True)
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
    parser.add_argument("--expected-track-sweeper-sha256", required=True)
    parser.add_argument("--expected-vfg-comparison-sha256", required=True)
    parser.add_argument("--expected-track-evaluator-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--expected-candidates", type=int, default=4)
    parser.add_argument("--schema-prefix", default="stage70-decoupled")
    parser.add_argument(
        "--passing-decision",
        default="passing pairs may proceed to paired export/backend validation without deployment",
    )
    parser.add_argument(
        "--failure-decision",
        default="reject every pair before test, backend, frozen video or deployment",
    )
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite pair validation evidence")
    if args.expected_candidates < 2:
        raise ValueError("pair validation requires at least two completed specialists")
    static_evaluator = args.scripts_root / "evaluate_attribute_baseline.py"
    track_sweeper = args.scripts_root / "sweep_attribute_track_fusion.py"
    vfg_comparison = args.scripts_root / "run_vfg7_validation_comparison.py"
    track_evaluator = args.scripts_root / "evaluate_attribute_track_fusion.py"
    pins = {
        "training_matrix": (args.training_matrix, args.expected_training_matrix_sha256),
        "hard_manifest": (args.hard_manifest, args.expected_hard_manifest_sha256),
        "ua_manifest": (args.ua_manifest, args.expected_ua_manifest_sha256),
        "vfg_manifest": (args.vfg_manifest, args.expected_vfg_manifest_sha256),
        "production_checkpoint": (args.production_checkpoint, args.expected_production_checkpoint_sha256),
        "labels": (args.labels, args.expected_labels_sha256),
        "static_evaluator": (static_evaluator, args.expected_static_evaluator_sha256),
        "track_sweeper": (track_sweeper, args.expected_track_sweeper_sha256),
        "vfg_comparison": (vfg_comparison, args.expected_vfg_comparison_sha256),
        "track_evaluator": (track_evaluator, args.expected_track_evaluator_sha256),
    }
    for name, (path, expected) in pins.items():
        require_sha(path, expected, name)
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": f"{args.schema_prefix}-pair-validation-v1", "status": "waiting_for_training",
        "created_at": now(), "updated_at": now(), "test_accessed": False,
        "frozen_video_used": False, "production_model_modified": False,
        "backend_gates_run": False, "deployment_performed": False,
        "deployment_paused_by_user": True,
        "expected_input_hashes": {name: expected.lower() for name, (_, expected) in pins.items()},
    }
    atomic_json(args.state, state)
    training = wait_for_training(args.training_state, args.training_session, args.timeout_hours)
    validate_training_contract(training, args.expected_training_matrix_sha256, args.expected_candidates)

    body_models: list[tuple[str, Path]] = []
    color_models: list[tuple[str, Path]] = []
    for record in training["completed"]:
        candidate = record["candidate_id"]
        root = args.training_root / candidate
        test_metrics = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
        if test_metrics.get("status") != "not_run":
            raise RuntimeError(f"candidate {candidate} unexpectedly accessed test")
        target = body_models if record.get("specialist") == "body" else color_models if record.get("specialist") == "color" else None
        if target is None:
            raise RuntimeError(f"candidate {candidate} has no specialist contract")
        for filename, suffix in (("best.pt", "best"), ("gate-best.pt", "gate-best")):
            checkpoint = root / filename
            if checkpoint.is_file():
                target.append((f"{candidate}-{suffix}", checkpoint.resolve()))
    if not body_models or not color_models:
        raise RuntimeError("training produced no body or color checkpoint")
    models = [("production", args.production_checkpoint.resolve()), *body_models, *color_models]
    hashes = {name: sha256(path) for name, path in models}
    state.update({
        "status": "running", "updated_at": now(), "training_state_sha256": sha256(args.training_state),
        "body_models": [name for name, _ in body_models], "color_models": [name for name, _ in color_models],
        "models": [{"name": name, "checkpoint": str(path), "sha256": hashes[name]} for name, path in models],
    })
    atomic_json(args.state, state)

    static: dict[str, dict] = {}
    for index, (name, checkpoint) in enumerate(models):
        state["active"] = {"stage": "hard_static", "model": name, "index": index + 1, "total": len(models)}
        state["updated_at"] = now(); atomic_json(args.state, state)
        slug = safe_name(name); output = args.output_root / f"{slug}.hard-validation.json"
        run([
            args.python, str(static_evaluator), "--manifest", str(args.hard_manifest),
            "--checkpoint", str(checkpoint), "--labels", str(args.labels), "--output", str(output),
            "--split", "validation", "--batch-size", "128", "--workers", "4", "--device", args.device,
            "--type-threshold", "0.75", "--color-threshold", "0.70",
            "--threshold-sweep", *[str(value) for value in THRESHOLDS],
        ], args.output_root / f"{slug}.hard-validation.log")
        report = json.loads(output.read_text(encoding="utf-8"))
        static[name] = {
            "body": select_threshold(report["threshold_sweep"]["body_type"]),
            "color": select_threshold(report["threshold_sweep"]["color"]),
            "report": str(output), "report_sha256": sha256(output),
        }

    ua: dict[str, dict] = {}
    ua_models = [("production", args.production_checkpoint.resolve()), *body_models]
    for index, (name, checkpoint) in enumerate(ua_models):
        state["active"] = {"stage": "ua_body_track", "model": name, "index": index + 1, "total": len(ua_models)}
        state["updated_at"] = now(); atomic_json(args.state, state)
        slug = safe_name(name); output = args.output_root / f"{slug}.ua-validation.json"
        run([
            args.python, str(track_sweeper), "--manifest", str(args.ua_manifest),
            "--checkpoint", str(checkpoint), "--output", str(output), "--split", "validation",
            "--minimum-window-frames", "3", "--thresholds", *[str(value) for value in THRESHOLDS],
            "--windows", "3", "5", "--minimum-shares", "0.50", "0.60", "0.70",
            "--minimum-margins", "0.05", "0.10", "0.15", "--batch-size", "128",
            "--workers", "4", "--device", args.device,
        ], args.output_root / f"{slug}.ua-validation.log")
        report = json.loads(output.read_text(encoding="utf-8"))
        ua[name] = {"body": compact_track(report["body_type"].get("selected")),
                    "report": str(output), "report_sha256": sha256(output)}

    state["active"] = {"stage": "vfg7_validation", "models": len(models)}
    state["updated_at"] = now(); atomic_json(args.state, state)
    vfg_root = args.output_root / "vfg7-comparison"
    command = [
        args.python, str(vfg_comparison), "--manifest", str(args.vfg_manifest),
        "--labels", str(args.labels), "--scripts-root", str(args.scripts_root),
        "--output-dir", str(vfg_root), "--python", args.python, "--batch-size", "128", "--workers", "4",
    ]
    for name, checkpoint in models:
        command.extend(["--model", f"{name}={checkpoint}"])
    run(command, args.output_root / "vfg7-comparison.log")
    vfg_report = vfg_root / "comparison-report.json"
    vfg = json.loads(vfg_report.read_text(encoding="utf-8"))
    vfg_by_name = {item["model"]: item for item in vfg["models"]}
    production_vfg = vfg_by_name["production"]
    baseline_body_coverage = static["production"]["body"]["coverage"]
    baseline_color_unknown = 1.0 - static["production"]["color"]["coverage"]

    pairs = []
    for body_name, body_checkpoint in body_models:
        for color_name, color_checkpoint in color_models:
            hard_body = static[body_name]["body"]; hard_color = static[color_name]["color"]
            ua_body = ua[body_name]["body"]
            body_gain = hard_body["coverage"] - baseline_body_coverage
            color_unknown = 1.0 - hard_color["coverage"]
            unknown_reduction = (
                (baseline_color_unknown - color_unknown) / baseline_color_unknown
                if baseline_color_unknown > 0 else 0.0
            )
            pair_vfg_gates, pair_vfg_comparison = vfg_pair_gates(
                vfg_by_name[body_name], vfg_by_name[color_name], production_vfg
            )
            gates = {
                "hard_body_precision_0_93": hard_body["precision"] >= 0.93,
                "hard_body_coverage_0_45": hard_body["coverage"] >= 0.45,
                "hard_color_precision_0_93": hard_color["precision"] >= 0.93,
                "hard_color_coverage_0_25": hard_color["coverage"] >= 0.25,
                "hard_body_coverage_gain_15pp": body_gain >= 0.15,
                "hard_color_unknown_reduction_20pct": unknown_reduction >= 0.20,
                "ua_body_precision_0_93": (ua_body.get("precision") or 0.0) >= 0.93,
                "ua_body_coverage_0_45": (ua_body.get("coverage") or 0.0) >= 0.45,
                "ua_body_stability_0_95": (ua_body.get("weighted_stability_rate") or 0.0) >= 0.95,
                "vfg_labeled_pair_screen": all(pair_vfg_gates.values()),
            }
            pairs.append({
                "pair_id": f"{body_name}__{color_name}",
                "body_model": body_name, "body_checkpoint": str(body_checkpoint),
                "body_checkpoint_sha256": hashes[body_name],
                "color_model": color_name, "color_checkpoint": str(color_checkpoint),
                "color_checkpoint_sha256": hashes[color_name],
                "hard": {"body_type": hard_body, "color": hard_color},
                "ua_track": {"body_type": ua_body},
                "vfg7": {
                    "body": {
                        "selected_threshold": vfg_by_name[body_name]["selected_thresholds"]["body_type"],
                        "track": vfg_by_name[body_name]["track_validation"],
                    },
                    "color": {
                        "selected_threshold": vfg_by_name[color_name]["selected_thresholds"]["color"],
                        "track": vfg_by_name[color_name]["track_validation"],
                    },
                    "gates": pair_vfg_gates, "comparison_to_production": pair_vfg_comparison,
                },
                "comparison_to_production": {
                    "hard_body_coverage_gain_percentage_points": body_gain * 100.0,
                    "hard_color_unknown_relative_reduction": unknown_reduction,
                },
                "gates": gates, "screen_status": "pass" if all(gates.values()) else "fail_closed",
            })

    for name, path in models:
        if sha256(path) != hashes[name]:
            raise RuntimeError(f"checkpoint changed during validation: {name}")
    for name, (path, expected) in pins.items():
        require_sha(path, expected, name)
    passing = [item["pair_id"] for item in pairs if item["screen_status"] == "pass"]
    report = {
        "schema_version": f"{args.schema_prefix}-pair-validation-screen-v1", "created_at": now(),
        "status": "pass_pairs_available" if passing else "complete_all_pairs_rejected",
        "training_state": str(args.training_state.resolve()), "training_state_sha256": sha256(args.training_state),
        "body_models": [name for name, _ in body_models], "color_models": [name for name, _ in color_models],
        "pairs": pairs, "passing_pairs": passing,
        "artifacts": {"vfg_comparison": str(vfg_report), "vfg_comparison_sha256": sha256(vfg_report)},
        "policy": {
            "validation_only": True, "decoupled_heads_combined_after_independent_inference": True,
            "body_metrics_only_from_body_specialist": True, "color_metrics_only_from_color_specialist": True,
            "test_accessed": False, "frozen_video_used": False, "production_model_modified": False,
            "backend_gates_run": False, "deployment_performed": False, "deployment_paused_by_user": True,
        },
        "decision": args.passing_decision if passing else args.failure_decision,
    }
    report_path = args.output_root / "pair-validation-screen-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.update({"status": "complete", "updated_at": now(), "active": None,
                  "report": str(report_path), "report_sha256": sha256(report_path), "passing_pairs": passing})
    atomic_json(args.state, state)
    print(json.dumps({"status": report["status"], "passing_pairs": passing,
                      "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
