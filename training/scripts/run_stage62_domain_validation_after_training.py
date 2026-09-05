#!/usr/bin/env python3
"""Wait for Stage62 training, then run validation-only hard/track/VFG screens."""

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
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86,
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


def normalize_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"invalid expected SHA256 for {label}: {value!r}")
    return normalized


def require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing immutable {label}: {path}")
    expected_normalized = normalize_sha256(expected, label)
    actual = sha256(path)
    if actual != expected_normalized:
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {actual} != {expected_normalized}")
    return actual


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"validation command failed ({result.returncode}); see {log}")


def wait_for_training(state_path: Path, session: str, timeout_hours: float) -> dict:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "complete_validation_only":
                return state
            if state.get("status") not in {"running", "complete_validation_only"}:
                raise RuntimeError(f"unexpected training state: {state.get('status')}")
        session_alive = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not session_alive:
            raise RuntimeError("training session ended without a complete state")
        time.sleep(30)
    raise TimeoutError("timed out waiting for Stage62 domain-consistency training")


def select_threshold(sweep: dict, precision_gate: float = 0.93) -> dict:
    rows = [
        {
            "threshold": float(threshold),
            "precision": float(metrics["high_confidence_precision"]),
            "coverage": float(metrics["high_confidence_coverage"]),
            "selected": int(metrics["high_confidence_selected"]),
        }
        for threshold, metrics in sweep.items()
    ]
    passing = [row for row in rows if row["precision"] >= precision_gate and row["selected"] > 0]
    if passing:
        return {
            **max(passing, key=lambda row: (row["coverage"], row["precision"], -row["threshold"])),
            "precision_gate_pass": True,
        }
    return {
        **max(rows, key=lambda row: (row["precision"], row["coverage"], row["threshold"])),
        "precision_gate_pass": False,
    }


def compact_track(selected: dict | None) -> dict:
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


def build_release_gates(
    *,
    hard_body: dict,
    hard_color: dict,
    ua_body: dict,
    vfg_screen_passed: bool,
    body_gain: float,
    color_unknown_reduction: float,
    is_baseline: bool,
) -> dict[str, bool]:
    """Build release gates only from datasets that contain truth for the attribute.

    UA-DETRAC supplies body truth but has no supervised color labels.  Color
    precision/coverage and color-track stability are therefore delegated to the
    labeled VFG validation screen instead of turning an absent label into a
    guaranteed zero-valued failure.
    """
    return {
        "hard_body_precision_0_93": hard_body["precision"] >= 0.93,
        "hard_body_coverage_0_45": hard_body["coverage"] >= 0.45,
        "hard_color_precision_0_93": hard_color["precision"] >= 0.93,
        "hard_color_coverage_0_25": hard_color["coverage"] >= 0.25,
        "hard_body_coverage_gain_15pp": is_baseline or body_gain >= 0.15,
        "hard_color_unknown_reduction_20pct": is_baseline or color_unknown_reduction >= 0.20,
        "ua_body_precision_0_93": (ua_body.get("precision") or 0.0) >= 0.93,
        "ua_body_coverage_0_45": (ua_body.get("coverage") or 0.0) >= 0.45,
        "ua_body_stability_0_95": (ua_body.get("weighted_stability_rate") or 0.0) >= 0.95,
        "vfg_labeled_body_color_and_track_screen": vfg_screen_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--training-matrix", type=Path)
    parser.add_argument("--expected-training-matrix-sha256")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-session", required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--expected-hard-manifest-sha256")
    parser.add_argument("--expected-ua-manifest-sha256")
    parser.add_argument("--expected-vfg-manifest-sha256")
    parser.add_argument("--expected-production-checkpoint-sha256")
    parser.add_argument("--expected-labels-sha256")
    parser.add_argument("--expected-static-evaluator-sha256")
    parser.add_argument("--expected-track-sweeper-sha256")
    parser.add_argument("--expected-vfg-comparison-sha256")
    parser.add_argument("--expected-track-evaluator-sha256")
    parser.add_argument(
        "--resume-waiting",
        action="store_true",
        help="Resume only a policy-clean waiting state whose output directory is still empty.",
    )
    args = parser.parse_args()
    static_evaluator = args.scripts_root / "evaluate_attribute_baseline.py"
    track_sweeper = args.scripts_root / "sweep_attribute_track_fusion.py"
    vfg_comparison = args.scripts_root / "run_vfg7_validation_comparison.py"
    track_evaluator = args.scripts_root / "evaluate_attribute_track_fusion.py"
    pinned_inputs = {
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
    pin_values = [path is not None and expected is not None for path, expected in pinned_inputs.values()]
    if any(pin_values) and not all(pin_values):
        raise RuntimeError("partial immutable-evidence pin set is forbidden")
    pinned_mode = all(pin_values)
    expected_input_hashes = {
        name: require_sha256(path, expected, name)
        for name, (path, expected) in pinned_inputs.items()
    } if pinned_mode else {}
    if args.output_root.exists() or args.state.exists():
        if not args.resume_waiting:
            raise FileExistsError("refusing to overwrite Stage62 validation evidence")
        if not args.output_root.is_dir() or not args.state.is_file():
            raise RuntimeError("resume requires both the existing output directory and state file")
        if any(args.output_root.iterdir()):
            raise RuntimeError("resume is allowed only before validation output has been written")
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if state.get("status") != "waiting_for_training":
            raise RuntimeError(f"resume requires waiting_for_training state, got {state.get('status')}")
        for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
            if state.get(key) is not False:
                raise RuntimeError(f"resume state violates fail-closed policy: {key}")
        state.update({
            "updated_at": now(),
            "resumed_waiter": True,
            "evaluator_policy_revision": "attribute-truth-routed-v2",
            "immutable_evidence_pinned": pinned_mode,
            "expected_input_hashes": expected_input_hashes,
        })
        atomic_json(args.state, state)
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "attribute-stage62-domain-validation-run-v1",
            "status": "waiting_for_training",
            "created_at": now(),
            "updated_at": now(),
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
            "evaluator_policy_revision": "attribute-truth-routed-v2",
            "immutable_evidence_pinned": pinned_mode,
            "expected_input_hashes": expected_input_hashes,
        }
        atomic_json(args.state, state)
    training = wait_for_training(args.training_state, args.training_session, args.timeout_hours)
    if pinned_mode:
        if training.get("matrix_sha256", "").lower() != expected_input_hashes["training_matrix"]:
            raise RuntimeError("training state is not bound to the approved matrix")
        if Path(training.get("matrix", "")).resolve() != args.training_matrix.resolve():
            raise RuntimeError("training state references an unexpected matrix path")
    if training.get("test_accessed") is not False or training.get("frozen_video_used") is not False:
        raise RuntimeError("training evidence violates test/frozen-video isolation")
    if training.get("production_model_modified") is not False or training.get("deployment_performed") is not False:
        raise RuntimeError("training evidence violates production isolation")
    if training.get("failed"):
        raise RuntimeError("one or more Stage62 candidates failed; preserve evidence before validation")

    models: list[tuple[str, Path]] = [("production", args.production_checkpoint.resolve())]
    for record in training.get("completed", []):
        candidate_id = record["candidate_id"]
        candidate_root = args.training_root / candidate_id
        test_metrics = json.loads((candidate_root / "test_metrics.json").read_text(encoding="utf-8"))
        if test_metrics.get("status") != "not_run":
            raise RuntimeError(f"candidate {candidate_id} unexpectedly accessed test")
        for filename, suffix in (("best.pt", "best"), ("gate-best.pt", "gate-best")):
            checkpoint = candidate_root / filename
            if checkpoint.is_file():
                models.append((f"{candidate_id}-{suffix}", checkpoint.resolve()))
    if len(models) <= 1:
        raise RuntimeError("no completed candidate checkpoints found")
    if len({name for name, _ in models}) != len(models):
        raise RuntimeError("validation model names are not unique")
    model_hashes = {name: sha256(path) for name, path in models}

    manifest_hashes = {
        "hard": sha256(args.hard_manifest),
        "ua": sha256(args.ua_manifest),
        "vfg": sha256(args.vfg_manifest),
    }
    state.update({
        "status": "running",
        "updated_at": now(),
        "training_state_sha256": sha256(args.training_state),
        "models": [{"name": name, "checkpoint": str(path), "sha256": model_hashes[name]} for name, path in models],
        "manifest_hashes": manifest_hashes,
    })
    atomic_json(args.state, state)
    threshold_args = [str(value) for value in THRESHOLDS]
    summaries = []
    for index, (name, checkpoint) in enumerate(models):
        slug = safe_name(name)
        state["active"] = {"model": name, "index": index + 1, "total": len(models)}
        state["updated_at"] = now()
        atomic_json(args.state, state)
        hard_output = args.output_root / f"{slug}.hard-validation.json"
        hard_log = args.output_root / f"{slug}.hard-validation.log"
        run(
            [
                args.python, str(static_evaluator),
                "--manifest", str(args.hard_manifest), "--checkpoint", str(checkpoint),
                "--labels", str(args.labels), "--output", str(hard_output),
                "--split", "validation", "--batch-size", "128", "--workers", "4",
                "--device", args.device, "--type-threshold", "0.75", "--color-threshold", "0.70",
                "--threshold-sweep", *threshold_args,
            ],
            hard_log,
        )
        ua_output = args.output_root / f"{slug}.ua-validation.json"
        ua_log = args.output_root / f"{slug}.ua-validation.log"
        run(
            [
                args.python, str(track_sweeper),
                "--manifest", str(args.ua_manifest), "--checkpoint", str(checkpoint),
                "--output", str(ua_output), "--split", "validation",
                "--minimum-window-frames", "3", "--thresholds", *threshold_args,
                "--windows", "3", "5", "--minimum-shares", "0.50", "0.60", "0.70",
                "--minimum-margins", "0.05", "0.10", "0.15",
                "--batch-size", "128", "--workers", "4", "--device", args.device,
            ],
            ua_log,
        )
        hard = json.loads(hard_output.read_text(encoding="utf-8"))
        ua = json.loads(ua_output.read_text(encoding="utf-8"))
        summaries.append({
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "hard": {
                "body_type": select_threshold(hard["threshold_sweep"]["body_type"]),
                "color": select_threshold(hard["threshold_sweep"]["color"]),
                "report": str(hard_output),
                "report_sha256": sha256(hard_output),
            },
            "ua_track": {
                "body_type": compact_track(ua["body_type"].get("selected")),
                "color": compact_track(ua["color"].get("selected")),
                "body_selection_status": ua["body_type"].get("selection_status"),
                "color_selection_status": ua["color"].get("selection_status"),
                "report": str(ua_output),
                "report_sha256": sha256(ua_output),
            },
        })

    vfg_root = args.output_root / "vfg7-comparison"
    vfg_log = args.output_root / "vfg7-comparison.log"
    vfg_command = [
        args.python, str(vfg_comparison),
        "--manifest", str(args.vfg_manifest), "--labels", str(args.labels),
        "--scripts-root", str(args.scripts_root), "--output-dir", str(vfg_root),
        "--python", args.python, "--batch-size", "128", "--workers", "4",
    ]
    for name, checkpoint in models:
        vfg_command.extend(["--model", f"{name}={checkpoint}"])
    run(vfg_command, vfg_log)
    vfg = json.loads((vfg_root / "comparison-report.json").read_text(encoding="utf-8"))
    vfg_by_name = {item["model"]: item for item in vfg["models"]}

    baseline = summaries[0]
    baseline_hard_body = baseline["hard"]["body_type"]["coverage"]
    baseline_hard_color_unknown = 1.0 - baseline["hard"]["color"]["coverage"]
    for item in summaries:
        hard_body = item["hard"]["body_type"]
        hard_color = item["hard"]["color"]
        ua_body = item["ua_track"]["body_type"]
        ua_color = item["ua_track"]["color"]
        vfg_item = vfg_by_name[item["model"]]
        body_gain = hard_body["coverage"] - baseline_hard_body
        color_unknown = 1.0 - hard_color["coverage"]
        color_unknown_reduction = (
            (baseline_hard_color_unknown - color_unknown) / baseline_hard_color_unknown
            if baseline_hard_color_unknown > 0 else 0.0
        )
        gates = build_release_gates(
            hard_body=hard_body,
            hard_color=hard_color,
            ua_body=ua_body,
            vfg_screen_passed=vfg_item.get("validation_screen_status") == "pass",
            body_gain=body_gain,
            color_unknown_reduction=color_unknown_reduction,
            is_baseline=item is baseline,
        )
        item["vfg7"] = {
            "selected_thresholds": vfg_item["selected_thresholds"],
            "track_validation": vfg_item["track_validation"],
            "validation_gates": vfg_item["validation_gates"],
            "validation_screen_status": vfg_item["validation_screen_status"],
        }
        item["comparison_to_production"] = {
            "hard_body_coverage_gain_percentage_points": body_gain * 100.0,
            "hard_color_unknown_relative_reduction": color_unknown_reduction,
        }
        item["ua_color_evidence_policy"] = {
            "release_gate": False,
            "reason": "UA-DETRAC validation manifest has no supervised color truth; predictions remain a diagnostic only",
            "evaluated": ua_color.get("evaluated"),
            "labeled_color_track_gate": "vfg_labeled_body_color_and_track_screen",
        }
        item["gates"] = gates
        item["screen_status"] = "pass" if all(gates.values()) else "fail_closed"

    if any(sha256(path) != manifest_hashes[key] for key, path in {
        "hard": args.hard_manifest, "ua": args.ua_manifest, "vfg": args.vfg_manifest,
    }.items()):
        raise RuntimeError("a validation manifest changed during evaluation")
    if pinned_mode:
        pinned_paths = {
            "training_matrix": args.training_matrix,
            "hard_manifest": args.hard_manifest,
            "ua_manifest": args.ua_manifest,
            "vfg_manifest": args.vfg_manifest,
            "production_checkpoint": args.production_checkpoint,
            "labels": args.labels,
            "static_evaluator": static_evaluator,
            "track_sweeper": track_sweeper,
            "vfg_comparison": vfg_comparison,
            "track_evaluator": track_evaluator,
        }
        if any(sha256(path) != expected_input_hashes[name] for name, path in pinned_paths.items()):
            raise RuntimeError("immutable validation evidence changed during evaluation")
        if sha256(args.training_state) != state["training_state_sha256"]:
            raise RuntimeError("training state changed during evaluation")
        if any(sha256(path) != model_hashes[name] for name, path in models):
            raise RuntimeError("a model checkpoint changed during evaluation")
    passing = [item["model"] for item in summaries[1:] if item["screen_status"] == "pass"]
    report = {
        "schema_version": "attribute-stage62-domain-validation-screen-v1",
        "created_at": now(),
        "status": "pass_candidates_available" if passing else "complete_all_candidates_rejected",
        "training_state": str(args.training_state.resolve()),
        "training_state_sha256": sha256(args.training_state),
        "manifest_hashes": manifest_hashes,
        "immutable_evidence_pinned": pinned_mode,
        "expected_input_hashes": expected_input_hashes,
        "models": summaries,
        "passing_candidates": passing,
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
            "attribute_truth_routing": {
                "ua_detrac": "body precision, coverage and stability only",
                "vfg7": "labeled body/color static and track precision, coverage and stability",
                "ua_color": "diagnostic only because evaluated color truth is zero",
            },
        },
        "decision": (
            "passing candidates may proceed to seed robustness before a single final test"
            if passing else
            "reject all candidates before test/backend/frozen-video/deployment and diagnose the next data or model change"
        ),
    }
    report_path = args.output_root / "validation-screen-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.update({
        "status": "complete",
        "updated_at": now(),
        "active": None,
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "passing_candidates": passing,
    })
    atomic_json(args.state, state)
    print(json.dumps({"status": report["status"], "passing_candidates": passing, "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
