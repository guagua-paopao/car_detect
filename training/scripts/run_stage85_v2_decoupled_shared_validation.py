#!/usr/bin/env python3
"""Run the validation-only joint gate after Stage80 and Stage84 finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WAITING = {
    "running",
    "running_migration",
    "running_training",
    "waiting_for_stage76_stage80",
    "waiting_for_stage77",
}


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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dependency_action(status: str) -> str:
    if status in WAITING:
        return "wait"
    if status == "complete_validation_only":
        return "ready"
    return "fail"


def select_checkpoint(state: dict[str, Any], role: str) -> tuple[Path, str]:
    if state.get("status") != "complete_validation_only":
        raise RuntimeError(f"{role} is not validation-only complete")
    path = Path(str(state.get("best_checkpoint") or ""))
    expected = str(state.get("best_checkpoint_sha256") or "").lower()
    if not path.is_file() or not expected or sha256(path).lower() != expected:
        raise RuntimeError(f"{role} checkpoint lineage mismatch")
    return path, expected


def validate_report_contract(evidence: dict[str, Any], body_sha: str, color_sha: str) -> None:
    policy = evidence.get("policy", {})
    inputs = evidence.get("inputs", {})
    rows = evidence.get("rows", {})
    comparison = evidence.get("comparison", {})
    gates = evidence.get("shared_validation_gates", {}).get("gates", {})
    required_gates = {
        "body_static_precision",
        "body_static_coverage",
        "body_complex_coverage_gain",
        "color_static_precision",
        "color_static_coverage",
        "color_complex_unknown_reduction",
        "body_track_precision",
        "body_track_coverage",
        "color_track_precision",
        "color_track_coverage",
        "body_track_stability",
        "color_track_stability",
    }
    if evidence.get("status") != "complete_validation_only":
        raise RuntimeError("Stage85 report status mismatch")
    if (
        policy.get("test_accessed") is not False
        or policy.get("frozen_video_used") is not False
        or policy.get("production_model_modified") is not False
        or policy.get("backend_gates_run") is not False
        or policy.get("deployment_performed") is not False
    ):
        raise RuntimeError("Stage85 report isolation contract mismatch")
    if (
        str(inputs.get("body_checkpoint_sha256") or "").lower() != body_sha.lower()
        or str(inputs.get("color_checkpoint_sha256") or "").lower() != color_sha.lower()
    ):
        raise RuntimeError("Stage85 report checkpoint lineage mismatch")
    if int(rows.get("complex_body_supervised") or 0) <= 0 or int(rows.get("complex_color_supervised") or 0) <= 0:
        raise RuntimeError("Stage85 complex subset evidence is missing")
    if not isinstance(evidence.get("complex_static"), dict) or not isinstance(evidence.get("per_class"), dict):
        raise RuntimeError("Stage85 complex or per-class report section is missing")
    if {
        "body_complex_static_coverage_gain",
        "color_complex_static_unknown_relative_reduction",
    } - set(comparison):
        raise RuntimeError("Stage85 complex comparison evidence is missing")
    if set(gates) != required_gates:
        raise RuntimeError("Stage85 gate contract mismatch")


def validate_matrix(matrix_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    matrix = read_json(matrix_path)
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage85 matrix is not prepared")
    if str(matrix.get("runner_sha256") or "").lower() != sha256(Path(__file__).resolve()).lower():
        raise RuntimeError("Stage85 runner SHA256 mismatch")
    policy = matrix.get("policy", {})
    required = {
        "split": "validation",
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise RuntimeError(f"Stage85 policy violation: {key}")
    paths: dict[str, Path] = {}
    for name, evidence in matrix.get("immutable_inputs", {}).items():
        path = Path(str(evidence.get("path") or ""))
        expected = str(evidence.get("sha256") or "").lower()
        if not path.is_file() or not expected or sha256(path).lower() != expected:
            raise RuntimeError(f"Stage85 immutable input mismatch: {name}")
        paths[name] = path
    required_names = {"evaluator", "manifest", "labels", "baseline_checkpoint"}
    if set(paths) != required_names:
        raise RuntimeError("Stage85 immutable input set mismatch")
    labels = read_json(paths["labels"])
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("Stage85 taxonomy-v2 labels mismatch")
    manifest_name = str(paths["manifest"]).lower()
    if "test" in manifest_name or "60s" in manifest_name or "36-48" in manifest_name:
        raise RuntimeError("Stage85 non-validation manifest path rejected")
    return matrix, paths


def wait_for_candidates(
    stage80_state: Path,
    stage84_state: Path,
    poll_seconds: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        color = read_json(stage80_state)
        body = read_json(stage84_state)
        color_action = dependency_action(str(color.get("status", "missing")))
        body_action = dependency_action(str(body.get("status", "missing")))
        if color_action == "fail":
            raise RuntimeError(f"Stage80 dependency failed closed: {color.get('status')}")
        if body_action == "fail":
            raise RuntimeError(f"Stage84 dependency failed closed: {body.get('status')}")
        if color_action == "ready" and body_action == "ready":
            time.sleep(5)
            return color, body
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage85 candidate wait timed out")
        time.sleep(max(5, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--stage80-state", type=Path, required=True)
    parser.add_argument("--stage84-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=259200)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix, paths = validate_matrix(args.matrix)
    for dependency in (args.stage80_state, args.stage84_state):
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
    if not args.datasets_safety_root.is_dir():
        raise FileNotFoundError(args.datasets_safety_root)
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only",
            "stage80_status": read_json(args.stage80_state).get("status"),
            "stage84_status": read_json(args.stage84_state).get("status"),
            "immutable_inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "test_accessed": False,
            "frozen_video_used": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage85 evidence")

    state: dict[str, Any] = {
        "schema_version": "stage85-v2-decoupled-shared-validation-run-v1",
        "status": "waiting_for_stage80_stage84",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    try:
        color_state, body_state = wait_for_candidates(
            args.stage80_state, args.stage84_state, args.poll_seconds, args.timeout_seconds
        )
        matrix, paths = validate_matrix(args.matrix)
        color_checkpoint, color_sha = select_checkpoint(color_state, "Stage80 color")
        body_checkpoint, body_sha = select_checkpoint(body_state, "Stage84 body")
        args.output_root.mkdir(parents=True, exist_ok=False)
        report = args.output_root / "stage85-v2-decoupled-shared-validation-report.json"
        log = args.output_root / "stage85-v2-decoupled-shared-validation.stdout.log"
        command = [
            sys.executable,
            str(paths["evaluator"]),
            "--manifest", str(paths["manifest"]),
            "--expected-manifest-sha256", sha256(paths["manifest"]),
            "--labels", str(paths["labels"]),
            "--expected-labels-sha256", sha256(paths["labels"]),
            "--body-checkpoint", str(body_checkpoint),
            "--expected-body-checkpoint-sha256", body_sha,
            "--color-checkpoint", str(color_checkpoint),
            "--expected-color-checkpoint-sha256", color_sha,
            "--baseline-checkpoint", str(paths["baseline_checkpoint"]),
            "--expected-baseline-checkpoint-sha256", sha256(paths["baseline_checkpoint"]),
            "--output", str(report),
            "--datasets-safety-root", str(args.datasets_safety_root.resolve()),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--precision-gate", str(matrix.get("precision_gate", 0.93)),
        ]
        state.update({
            "status": "running_validation",
            "updated_at": now(),
            "body_checkpoint": str(body_checkpoint),
            "body_checkpoint_sha256": body_sha,
            "color_checkpoint": str(color_checkpoint),
            "color_checkpoint_sha256": color_sha,
            "command": command,
        })
        atomic_json(args.state, state)
        with log.open("wb") as handle:
            result = subprocess.run(
                command,
                cwd=str(paths["evaluator"].resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not report.is_file():
            raise RuntimeError(f"Stage85 evaluator failed with return code {result.returncode}")
        evidence = read_json(report)
        validate_report_contract(evidence, body_sha, color_sha)
        passed = evidence.get("shared_validation_gates", {}).get("all_pass") is True
        state.update({
            "status": (
                "complete_validation_only_shared_pass_pending_evidence"
                if passed else "rejected_validation_gate_failure"
            ),
            "updated_at": now(),
            "return_code": result.returncode,
            "stdout_log": str(log.resolve()),
            "stdout_log_sha256": sha256(log),
            "report": str(report.resolve()),
            "report_sha256": sha256(report),
            "shared_validation_gates_pass": passed,
            "decision": evidence.get("shared_validation_gates", {}).get("decision"),
        })
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        state.update({
            "status": "failed_closed_runtime",
            "updated_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
