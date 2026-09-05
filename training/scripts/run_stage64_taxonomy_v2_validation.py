#!/usr/bin/env python3
"""Validate taxonomy-v2 candidates on exact validation-only evidence."""
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


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"validation failed ({result.returncode}); see {log}")


def wait_for_training(path: Path, session: str, timeout_hours: float) -> dict:
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "complete_validation_only":
                return state
            if state.get("status") != "running":
                raise RuntimeError(f"unexpected Stage64 training status: {state.get('status')}")
        alive = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
        if not alive:
            raise RuntimeError("Stage64 training session ended without complete evidence")
        time.sleep(30)
    raise TimeoutError("timed out waiting for Stage64 training")


def select_threshold(sweep: dict, precision_gate: float = 0.93) -> dict:
    rows = [{
        "threshold": float(threshold),
        "precision": float(metrics["high_confidence_precision"]),
        "coverage": float(metrics["high_confidence_coverage"]),
        "selected": int(metrics["high_confidence_selected"]),
        "predicted_unknown_rate": float(metrics["predicted_unknown_rate"]),
    } for threshold, metrics in sweep.items()]
    passing = [row for row in rows if row["precision"] >= precision_gate and row["selected"] > 0]
    selected = max(
        passing or rows,
        key=(lambda row: (row["coverage"], row["precision"], -row["threshold"])) if passing
        else (lambda row: (row["precision"], row["coverage"], row["threshold"])),
    )
    return {**selected, "precision_gate_pass": bool(passing)}


def compact_track(report: dict, head: str) -> dict:
    selected = report[head].get("selected")
    if not selected:
        return {"selection_status": report[head].get("selection_status")}
    final = selected["final_window"]
    stability = selected["stability"]
    return {
        "selection_status": report[head].get("selection_status"),
        "threshold": selected["threshold"],
        "window": selected["window"],
        "precision": final.get("precision"),
        "coverage": final.get("coverage"),
        "effective_unknown_rate": final.get("effective_unknown_rate"),
        "weighted_stability_rate": stability.get("weighted_stability_rate"),
        "label_switches_total": stability.get("label_switches_total"),
    }


def validate_views(report_path: Path, labels: Path) -> dict[str, Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("labels_sha256", "").lower() != sha256(labels):
        raise RuntimeError("Stage64 validation-view evidence is invalid")
    policy = report.get("policy", {})
    for key in ("validation_only", "merged_or_unsupported_truth_demoted_not_guessed", "exact_gray_silver_truth_preserved_when_present"):
        if policy.get(key) is not True:
            raise RuntimeError(f"Stage64 validation-view policy violation: {key}")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage64 validation-view policy violation: {key}")
    views = {}
    for item in report.get("sources", []):
        path = Path(item["output"])
        if not path.is_file() or sha256(path) != item["output_sha256"].lower():
            raise RuntimeError(f"Stage64 validation view changed: {item.get('name')}")
        views[item["name"]] = path
    if set(views) != {"hard", "ua", "vfg"}:
        raise RuntimeError("Stage64 validation views are incomplete")
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--expected-training-matrix-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-session", required=True)
    parser.add_argument("--taxonomy-manifest", type=Path, required=True)
    parser.add_argument("--views-report", type=Path, required=True)
    parser.add_argument("--production-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-production-checkpoint-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--expected-taxonomy-manifest-sha256", required=True)
    parser.add_argument("--expected-views-report-sha256", required=True)
    parser.add_argument("--expected-static-evaluator-sha256", required=True)
    parser.add_argument("--expected-track-evaluator-sha256", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    parser.add_argument("--resume-waiting", action="store_true")
    args = parser.parse_args()

    static_evaluator = args.scripts_root / "evaluate_attribute_baseline.py"
    track_evaluator = args.scripts_root / "sweep_attribute_track_fusion.py"
    expected_inputs = {
        "training_matrix": (args.training_matrix, args.expected_training_matrix_sha256),
        "taxonomy_manifest": (args.taxonomy_manifest, args.expected_taxonomy_manifest_sha256),
        "views_report": (args.views_report, args.expected_views_report_sha256),
        "production_checkpoint": (args.production_checkpoint, args.expected_production_checkpoint_sha256),
        "labels": (args.labels, args.expected_labels_sha256),
        "static_evaluator": (static_evaluator, args.expected_static_evaluator_sha256),
        "track_evaluator": (track_evaluator, args.expected_track_evaluator_sha256),
    }
    expected_input_hashes = {
        name: require_sha256(path, expected, name)
        for name, (path, expected) in expected_inputs.items()
    }
    resumed_created_at = None
    if args.output_root.exists() or args.state.exists():
        if not args.resume_waiting or not args.output_root.is_dir() or not args.state.is_file():
            raise FileExistsError("refusing to overwrite Stage64 validation evidence")
        if any(args.output_root.iterdir()):
            raise RuntimeError("cannot resume Stage64 validation: output directory is not empty")
        previous = json.loads(args.state.read_text(encoding="utf-8"))
        required_false = ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed")
        if previous.get("status") != "waiting_for_training" or any(previous.get(key) is not False for key in required_false):
            raise RuntimeError("cannot resume Stage64 validation: existing state is not a clean waiting state")
        resumed_created_at = previous.get("created_at")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
        args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "stage64-taxonomy-v2-validation-run-v1",
        "status": "waiting_for_training", "created_at": resumed_created_at or now(), "updated_at": now(),
        "resumed_with_pinned_evidence": resumed_created_at is not None,
        "expected_input_hashes": expected_input_hashes,
        "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    training = wait_for_training(args.training_state, args.training_session, args.timeout_hours)
    if training.get("matrix_sha256", "").lower() != expected_input_hashes["training_matrix"]:
        raise RuntimeError("Stage64 training state is not bound to the approved matrix")
    if Path(training.get("matrix", "")).resolve() != args.training_matrix.resolve():
        raise RuntimeError("Stage64 training state references an unexpected matrix path")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if training.get(key) is not False:
            raise RuntimeError(f"Stage64 training policy violation: {key}")
    if training.get("failed"):
        raise RuntimeError("a Stage64 candidate failed; validation stops fail closed")
    views = validate_views(args.views_report, args.labels)
    models = []
    for record in training.get("completed", []):
        candidate_id = record["candidate_id"]
        root = args.training_root / candidate_id
        test_metrics = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
        if test_metrics.get("status") != "not_run":
            raise RuntimeError(f"{candidate_id} unexpectedly accessed test")
        for filename, suffix in (("best.pt", "best"), ("gate-best.pt", "gate-best")):
            checkpoint = root / filename
            if checkpoint.is_file():
                models.append((f"{candidate_id}-{suffix}", checkpoint.resolve()))
    if not models:
        raise RuntimeError("no Stage64 candidate checkpoints")
    all_models = [("production", args.production_checkpoint.resolve()), *models]
    evidence_paths = {
        "training_matrix": args.training_matrix,
        "training_state": args.training_state,
        "taxonomy": args.taxonomy_manifest,
        "views_report": args.views_report,
        "labels": args.labels,
        "production_checkpoint": args.production_checkpoint,
        "static_evaluator": static_evaluator,
        "track_evaluator": track_evaluator,
        **views,
    }
    evidence_hashes = {key: sha256(path) for key, path in evidence_paths.items()}
    state.update({
        "status": "running", "updated_at": now(), "training_state_sha256": sha256(args.training_state),
        "models": [{"name": name, "checkpoint": str(path), "sha256": sha256(path)} for name, path in all_models],
        "evidence_hashes": evidence_hashes,
    })
    atomic_json(args.state, state)
    thresholds = [str(value) for value in THRESHOLDS]

    def static_eval(name: str, checkpoint: Path, dataset: str, manifest: Path) -> dict:
        slug = safe_name(name)
        output = args.output_root / f"{slug}.{dataset}.json"
        run([
            args.python, str(static_evaluator),
            "--manifest", str(manifest), "--checkpoint", str(checkpoint),
            "--labels", str(args.labels), "--output", str(output), "--split", "validation",
            "--batch-size", "128", "--workers", "4", "--device", args.device,
            "--type-threshold", "0.75", "--color-threshold", "0.70", "--threshold-sweep", *thresholds,
        ], args.output_root / f"{slug}.{dataset}.log")
        payload = json.loads(output.read_text(encoding="utf-8"))
        return {
            "body_type": select_threshold(payload["threshold_sweep"]["body_type"]),
            "color": select_threshold(payload["threshold_sweep"]["color"]),
            "report": str(output), "report_sha256": sha256(output),
        }

    def track_eval(name: str, checkpoint: Path, dataset: str, manifest: Path) -> dict:
        slug = safe_name(name)
        output = args.output_root / f"{slug}.{dataset}-track.json"
        run([
            args.python, str(track_evaluator),
            "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--output", str(output),
            "--split", "validation", "--minimum-window-frames", "3", "--thresholds", *thresholds,
            "--windows", "3", "5", "--minimum-shares", "0.50", "0.60", "0.70",
            "--minimum-margins", "0.05", "0.10", "0.15", "--batch-size", "128",
            "--workers", "4", "--device", args.device,
        ], args.output_root / f"{slug}.{dataset}-track.log")
        payload = json.loads(output.read_text(encoding="utf-8"))
        return {
            "body_type": compact_track(payload, "body_type"), "color": compact_track(payload, "color"),
            "report": str(output), "report_sha256": sha256(output),
        }

    summaries = []
    for index, (name, checkpoint) in enumerate(all_models):
        state["active"] = {"model": name, "index": index + 1, "total": len(all_models)}
        state["updated_at"] = now()
        atomic_json(args.state, state)
        item = {
            "model": name, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "hard": static_eval(name, checkpoint, "hard", views["hard"]),
            "vfg": static_eval(name, checkpoint, "vfg", views["vfg"]),
            "ua_track": track_eval(name, checkpoint, "ua", views["ua"]),
            "vfg_track": track_eval(name, checkpoint, "vfg", views["vfg"]),
        }
        if name != "production":
            item["taxonomy_exact"] = static_eval(name, checkpoint, "taxonomy", args.taxonomy_manifest)
        summaries.append(item)

    baseline = summaries[0]
    base_hard_body_coverage = baseline["hard"]["body_type"]["coverage"]
    base_vfg_color_unknown = 1.0 - baseline["vfg"]["color"]["coverage"]
    passing = []
    for item in summaries[1:]:
        exact = item["taxonomy_exact"]
        hard_body = item["hard"]["body_type"]
        vfg_color = item["vfg"]["color"]
        ua_body = item["ua_track"]["body_type"]
        vfg_track_color = item["vfg_track"]["color"]
        body_gain = hard_body["coverage"] - base_hard_body_coverage
        color_unknown = 1.0 - vfg_color["coverage"]
        color_reduction = ((base_vfg_color_unknown - color_unknown) / base_vfg_color_unknown) if base_vfg_color_unknown > 0 else 0.0
        gates = {
            "taxonomy_exact_body_precision_0_93": exact["body_type"]["precision"] >= 0.93,
            "taxonomy_exact_body_coverage_0_45": exact["body_type"]["coverage"] >= 0.45,
            "taxonomy_exact_color_precision_0_93": exact["color"]["precision"] >= 0.93,
            "taxonomy_exact_color_coverage_0_25": exact["color"]["coverage"] >= 0.25,
            "hard_exact_body_precision_0_93": hard_body["precision"] >= 0.93,
            "hard_exact_body_coverage_0_45": hard_body["coverage"] >= 0.45,
            "hard_exact_body_gain_15pp": body_gain >= 0.15,
            "vfg_shared_color_precision_0_93": vfg_color["precision"] >= 0.93,
            "vfg_shared_color_coverage_0_25": vfg_color["coverage"] >= 0.25,
            "vfg_shared_color_unknown_reduction_20pct": color_reduction >= 0.20,
            "ua_track_body_precision_0_93": (ua_body.get("precision") or 0.0) >= 0.93,
            "ua_track_body_coverage_0_45": (ua_body.get("coverage") or 0.0) >= 0.45,
            "ua_track_body_stability_0_95": (ua_body.get("weighted_stability_rate") or 0.0) >= 0.95,
            "vfg_track_color_precision_0_93": (vfg_track_color.get("precision") or 0.0) >= 0.93,
            "vfg_track_color_coverage_0_25": (vfg_track_color.get("coverage") or 0.0) >= 0.25,
            "vfg_track_color_stability_0_95": (vfg_track_color.get("weighted_stability_rate") or 0.0) >= 0.95,
        }
        item["comparison_to_production"] = {
            "hard_exact_body_coverage_gain_percentage_points": body_gain * 100,
            "vfg_shared_color_unknown_relative_reduction": color_reduction,
        }
        item["gates"] = gates
        item["screen_status"] = "pass" if all(gates.values()) else "fail_closed"
        if item["screen_status"] == "pass":
            passing.append(item["model"])
    immutable_models = {f"model:{name}": checkpoint for name, checkpoint in all_models}
    immutable_hashes = {key: sha256(path) for key, path in immutable_models.items()}
    if any(sha256(path) != evidence_hashes[key] for key, path in evidence_paths.items()):
        raise RuntimeError("Stage64 validation evidence changed during evaluation")
    if any(sha256(path) != immutable_hashes[key] for key, path in immutable_models.items()):
        raise RuntimeError("Stage64 model checkpoint changed during evaluation")
    report = {
        "schema_version": "stage64-taxonomy-v2-validation-screen-v1", "created_at": now(),
        "status": "pass_candidates_available" if passing else "complete_all_candidates_rejected",
        "models": summaries, "passing_candidates": passing, "evidence_hashes": evidence_hashes,
        "metric_policy": {
            "exact_taxonomy_metrics_are_gates": True,
            "truck_family_metric_may_be_reported_by_training_but_never_replaces_exact_gate": True,
        },
        "policy": {
            "validation_only": True, "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "backend_gates_run": False,
            "deployment_performed": False, "deployment_paused_by_user": True,
        },
    }
    report_path = args.output_root / "validation-screen-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.update({
        "status": "complete", "updated_at": now(), "active": None,
        "report": str(report_path), "report_sha256": sha256(report_path), "passing_candidates": passing,
    })
    atomic_json(args.state, state)
    print(json.dumps({"status": report["status"], "passing_candidates": passing, "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
