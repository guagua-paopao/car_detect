#!/usr/bin/env python3
"""Queue secondary hard/weather validation after Stage85 primary gates pass."""

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


WAITING = {"waiting_for_stage80_stage84", "running_validation"}


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


def primary_action(status: str) -> str:
    if status in WAITING:
        return "wait"
    if status == "complete_validation_only_shared_pass_pending_evidence":
        return "ready"
    if status in {
        "rejected_validation_gate_failure",
        "failed_closed_runtime",
        "failed_closed_validation",
    }:
        return "reject"
    return "fail"


def validate_matrix(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    matrix = read_json(path)
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage86 matrix is not prepared")
    if str(matrix.get("runner_sha256") or "").lower() != sha256(Path(__file__).resolve()).lower():
        raise RuntimeError("Stage86 runner SHA256 mismatch")
    policy = matrix.get("policy", {})
    for key in (
        "test_accessed", "frozen_video_used", "production_model_modified",
        "backend_gates_run", "deployment_performed",
    ):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage86 policy violation: {key}")
    if policy.get("threshold_tuning_on_secondary_views") is not False:
        raise RuntimeError("Stage86 threshold-freeze policy violation")
    paths: dict[str, Path] = {}
    for name, evidence in matrix.get("immutable_inputs", {}).items():
        item = Path(str(evidence.get("path") or ""))
        expected = str(evidence.get("sha256") or "").lower()
        if not item.is_file() or not expected or sha256(item).lower() != expected:
            raise RuntimeError(f"Stage86 immutable input mismatch: {name}")
        paths[name] = item
    required = {"evaluator", "primary_evaluator", "hard_manifest", "weather_manifest", "labels", "baseline_checkpoint"}
    if set(paths) != required:
        raise RuntimeError("Stage86 immutable input set mismatch")
    return matrix, paths


def wait_for_primary(path: Path, poll_seconds: int, timeout_seconds: int) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        value = read_json(path)
        action = primary_action(str(value.get("status", "missing")))
        if action in {"ready", "reject"}:
            return action, value
        if action == "fail":
            raise RuntimeError(f"unexpected Stage85 status: {value.get('status')}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage86 primary wait timed out")
        time.sleep(max(5, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--primary-state", type=Path, required=True)
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
    if not args.primary_state.is_file() or not args.datasets_safety_root.is_dir():
        raise FileNotFoundError("Stage86 dependency or datasets root is missing")
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only",
            "primary_status": read_json(args.primary_state).get("status"),
            "test_accessed": False,
            "frozen_video_used": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage86 evidence")
    state: dict[str, Any] = {
        "schema_version": "stage86-secondary-body-validation-run-v1",
        "status": "waiting_for_stage85_primary",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    atomic_json(args.state, state)
    try:
        action, primary_state = wait_for_primary(
            args.primary_state, args.poll_seconds, args.timeout_seconds
        )
        if action == "reject":
            state.update({
                "status": "not_run_primary_validation_rejected",
                "updated_at": now(),
                "primary_status": primary_state.get("status"),
                "reason": "secondary thresholds are unavailable because the primary candidate was rejected",
            })
            atomic_json(args.state, state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        matrix, paths = validate_matrix(args.matrix)
        primary_report = Path(str(primary_state.get("report") or ""))
        primary_report_sha = str(primary_state.get("report_sha256") or "").lower()
        if not primary_report.is_file() or not primary_report_sha or sha256(primary_report).lower() != primary_report_sha:
            raise RuntimeError("Stage85 primary report lineage mismatch")
        primary = read_json(primary_report)
        inputs = primary.get("inputs", {})
        body = Path(str(inputs.get("body_checkpoint") or ""))
        color = Path(str(inputs.get("color_checkpoint") or ""))
        body_sha = str(inputs.get("body_checkpoint_sha256") or "").lower()
        color_sha = str(inputs.get("color_checkpoint_sha256") or "").lower()
        if not body.is_file() or sha256(body).lower() != body_sha:
            raise RuntimeError("Stage86 body checkpoint lineage mismatch")
        if not color.is_file() or sha256(color).lower() != color_sha:
            raise RuntimeError("Stage86 color checkpoint lineage mismatch")
        if str(inputs.get("baseline_checkpoint_sha256") or "").lower() != sha256(paths["baseline_checkpoint"]):
            raise RuntimeError("Stage86 baseline lineage mismatch")
        args.output_root.mkdir(parents=True, exist_ok=False)
        report = args.output_root / "stage86-secondary-body-validation-report.json"
        log = args.output_root / "stage86-secondary-body-validation.stdout.log"
        command = [
            sys.executable, str(paths["evaluator"]),
            "--primary-evaluator", str(paths["primary_evaluator"]),
            "--expected-primary-evaluator-sha256", sha256(paths["primary_evaluator"]),
            "--primary-report", str(primary_report),
            "--expected-primary-report-sha256", primary_report_sha,
            "--hard-manifest", str(paths["hard_manifest"]),
            "--expected-hard-manifest-sha256", sha256(paths["hard_manifest"]),
            "--weather-manifest", str(paths["weather_manifest"]),
            "--expected-weather-manifest-sha256", sha256(paths["weather_manifest"]),
            "--labels", str(paths["labels"]),
            "--expected-labels-sha256", sha256(paths["labels"]),
            "--body-checkpoint", str(body),
            "--expected-body-checkpoint-sha256", body_sha,
            "--color-checkpoint", str(color),
            "--expected-color-checkpoint-sha256", color_sha,
            "--baseline-checkpoint", str(paths["baseline_checkpoint"]),
            "--expected-baseline-checkpoint-sha256", sha256(paths["baseline_checkpoint"]),
            "--datasets-safety-root", str(args.datasets_safety_root.resolve()),
            "--output", str(report),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
        ]
        state.update({"status": "running_secondary_validation", "updated_at": now(), "command": command})
        atomic_json(args.state, state)
        with log.open("wb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not report.is_file():
            raise RuntimeError(f"Stage86 evaluator failed with return code {result.returncode}")
        evidence = read_json(report)
        policy = evidence.get("policy", {})
        if (
            evidence.get("status") != "complete_validation_only"
            or policy.get("test_accessed") is not False
            or policy.get("frozen_video_used") is not False
        ):
            raise RuntimeError("Stage86 report contract mismatch")
        passed = evidence.get("secondary_gates", {}).get("all_pass") is True
        state.update({
            "status": "complete_secondary_validation_pass" if passed else "rejected_secondary_validation_gate_failure",
            "updated_at": now(),
            "return_code": result.returncode,
            "stdout_log": str(log.resolve()),
            "stdout_log_sha256": sha256(log),
            "report": str(report.resolve()),
            "report_sha256": sha256(report),
            "secondary_gates_pass": passed,
            "decision": evidence.get("secondary_gates", {}).get("decision"),
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
