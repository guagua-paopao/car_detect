#!/usr/bin/env python3
"""Queue one isolated taxonomy-v2 color teacher after Stage77 releases the GPU."""

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


META_KEYS = {"candidate_id", "purpose", "init_checkpoint_sha256"}


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


def command_arguments(parameters: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in parameters.items():
        if key in META_KEYS or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def stage77_action(status: str) -> str:
    if status in {"waiting_for_stage76", "running_validation"}:
        return "wait"
    if status == "complete_validation_only":
        return "ready"
    return "fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--stage77-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_matrix(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path]]:
    matrix = read_json(args.matrix)
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage80 matrix is not prepared")
    execution = matrix.get("execution_policy", {})
    required = {
        "iterative_test_access": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "research_only": True,
    }
    for key, expected in required.items():
        if execution.get(key) is not expected:
            raise RuntimeError(f"Stage80 execution policy violation: {key}")
    if execution.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage80 lost --skip-test")
    if sha256(Path(__file__).resolve()).lower() != str(
        matrix.get("runner_sha256", "")
    ).lower():
        raise RuntimeError("Stage80 runner SHA256 mismatch")

    paths: dict[str, Path] = {}
    for name, evidence in matrix.get("immutable_inputs", {}).items():
        path = Path(evidence["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path).lower() != str(evidence["sha256"]).lower():
            raise RuntimeError(f"Stage80 immutable input SHA256 mismatch: {name}")
        paths[name] = path
    labels = read_json(paths["labels"])
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("Stage80 requires isolated taxonomy-v2 labels")
    manifest_report = read_json(paths["manifest_report"])
    if manifest_report.get("status") != "pass":
        raise RuntimeError("Stage80 manifest report did not pass")
    policy = manifest_report.get("policy", {})
    for key in ("test_rows_imported", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage80 manifest policy violation: {key}")
    if manifest_report.get("output", {}).get("sha256", "").lower() != sha256(
        paths["manifest"]
    ):
        raise RuntimeError("Stage80 manifest/report lineage mismatch")
    migration = read_json(paths["migration_report"])
    if migration.get("status") != "pass":
        raise RuntimeError("Stage80 migration report did not pass")
    if migration.get("output_checkpoint_sha256", "").lower() != sha256(
        paths["init_checkpoint"]
    ):
        raise RuntimeError("Stage80 migration/checkpoint lineage mismatch")
    return matrix, paths


def wait_for_stage77(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        value = read_json(args.stage77_state)
        action = stage77_action(str(value.get("status", "missing")))
        if action == "ready":
            return value
        if action == "fail":
            raise RuntimeError(
                f"Stage77 did not complete cleanly: {value.get('status', 'missing')}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage80 wait for Stage77 timed out")
        time.sleep(max(5, args.poll_seconds))


def main() -> int:
    args = parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage80 evidence")
    matrix, paths = validate_matrix(args)
    if not args.stage77_state.is_file():
        raise FileNotFoundError(args.stage77_state)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "pass_preflight_only",
                    "stage77_status": read_json(args.stage77_state).get("status"),
                    "test_accessed": False,
                    "frozen_video_used": False,
                },
                indent=2,
            )
        )
        return 0

    state: dict[str, Any] = {
        "schema_version": "stage80-v2-color-teacher-run-v1",
        "status": "waiting_for_stage77",
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
        "eligibility": "research-only_non-deployable",
    }
    atomic_json(args.state, state)
    try:
        stage77 = wait_for_stage77(args)
        args.output_root.mkdir(parents=True, exist_ok=False)
        parameters = dict(matrix["parameters"])
        parameters.update(
            {
                "manifest": str(paths["manifest"].resolve()),
                "labels": str(paths["labels"].resolve()),
                "init_checkpoint": str(paths["init_checkpoint"].resolve()),
                "device": args.device,
                "code_revision": args.code_revision,
                "output_dir": str(args.output_root.resolve()),
            }
        )
        command = [
            sys.executable,
            str(paths["training_script"].resolve()),
            *command_arguments(parameters),
        ]
        if "--skip-test" not in command:
            raise RuntimeError("Stage80 training command lost --skip-test")
        state.update(
            {
                "status": "running",
                "updated_at": now(),
                "stage77_decision": stage77.get("decision"),
                "command": command,
            }
        )
        atomic_json(args.state, state)
        log = args.output_root / "stage80-v2-color-teacher.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(
                command,
                cwd=str(paths["training_script"].resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        best = args.output_root / "best.pt"
        state.update(
            {
                "status": (
                    "complete_validation_only"
                    if result.returncode == 0 and best.is_file()
                    else "failed_closed_training"
                ),
                "updated_at": now(),
                "return_code": result.returncode,
                "stdout_log": str(log.resolve()),
                "stdout_log_sha256": sha256(log),
                "best_checkpoint": str(best.resolve()) if best.is_file() else None,
                "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
            }
        )
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state["status"] == "complete_validation_only" else 2
    except Exception as error:
        state.update(
            {
                "status": "failed_closed_runtime",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
