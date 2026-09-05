#!/usr/bin/env python3
"""Recover an interrupted Stage71 teacher matrix without reopening test data.

The original Stage71 runner intentionally refuses an existing output directory.
This recovery entry point is narrower: it accepts only the SHA-pinned original
runner/matrix/state combination, resumes the single active candidate from its
``last.pt``, and then continues the remaining candidate in matrix order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCKED_FALSE_FLAGS = (
    "test_accessed",
    "frozen_video_used",
    "production_model_modified",
    "backend_gates_run",
    "deployment_performed",
)


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


def load_original_runner(path: Path):
    spec = importlib.util.spec_from_file_location("stage71_original_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original Stage71 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_locked_flags(state: dict[str, Any]) -> None:
    for key in LOCKED_FALSE_FLAGS:
        if state.get(key) is not False:
            raise RuntimeError(f"Stage71 recovery state violates locked flag: {key}")
    if state.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage71 recovery lost the deployment hold")


def replace_init_with_resume(command: list[str], checkpoint: Path) -> list[str]:
    result: list[str] = []
    index = 0
    removed = 0
    while index < len(command):
        if command[index] == "--init-checkpoint":
            if index + 1 >= len(command):
                raise RuntimeError("malformed --init-checkpoint in original command")
            removed += 1
            index += 2
            continue
        if command[index] == "--resume":
            raise RuntimeError("original command already contains --resume")
        result.append(command[index])
        index += 1
    if removed != 1:
        raise RuntimeError("original command must contain exactly one --init-checkpoint")
    result.extend(["--resume", str(checkpoint.resolve())])
    if "--skip-test" not in result:
        raise RuntimeError("recovery command lost --skip-test")
    return result


def training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid training log JSON at {path}:{line_number}"
                ) from error
    return rows


def checkpoint_metadata(
    path: Path,
    *,
    expected_architecture: str,
    expected_input_size: int,
    expected_selection_head: str,
    expected_epochs: int,
) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"recovery checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("recovery checkpoint is not a dictionary")
    architecture = str(checkpoint.get("architecture", ""))
    if architecture not in {
        expected_architecture,
        f"{expected_architecture}_multitask",
    }:
        raise RuntimeError("recovery checkpoint architecture mismatch")
    if int(checkpoint.get("input_size", -1)) != expected_input_size:
        raise RuntimeError("recovery checkpoint input-size mismatch")
    if checkpoint.get("selection_head") != expected_selection_head:
        raise RuntimeError("recovery checkpoint specialist-head mismatch")
    for key in ("model_state", "optimizer_state", "scheduler_state"):
        if key not in checkpoint:
            raise RuntimeError(f"recovery checkpoint lacks {key}")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 0 or epoch >= expected_epochs:
        raise RuntimeError(f"recovery checkpoint epoch is outside [0,{expected_epochs - 1}]")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "epoch": epoch,
        "best_score": float(checkpoint.get("best_score", -1.0)),
        "architecture": architecture,
        "input_size": int(checkpoint["input_size"]),
        "selection_head": checkpoint["selection_head"],
        "optimizer_state_present": checkpoint.get("optimizer_state") is not None,
        "scheduler_state_present": checkpoint.get("scheduler_state") is not None,
    }


def candidate_parameters(
    matrix: dict[str, Any], candidate: dict[str, Any], inputs: dict[str, Path], device: str, code_revision: str, output_dir: Path
) -> dict[str, Any]:
    specialist = candidate["specialist"]
    manifest_key = "body_manifest" if specialist == "body" else "color_manifest"
    return {
        **matrix["common"],
        **candidate,
        "manifest": str(inputs[manifest_key].resolve()),
        "unlabeled_manifest": str(inputs["unlabeled_manifest"].resolve()),
        "unlabeled_root": str(Path(matrix["datasets_safety_root"]).resolve()),
        "labels": str(inputs["labels"].resolve()),
        "device": device,
        "dataset_version": "attribute-domain-v2-stage71-convnext-teachers-research-v1",
        "code_revision": code_revision,
        "output_dir": str(output_dir.resolve()),
    }


def original_command(module, training_script: Path, parameters: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(training_script.resolve()),
        *module.command_arguments(parameters),
    ]


def verify_completed_candidate(
    output_dir: Path, candidate: dict[str, Any], matrix: dict[str, Any]
) -> dict[str, Any]:
    best = output_dir / "best.pt"
    last = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.json"
    test_metrics_path = output_dir / "test_metrics.json"
    model_card_path = output_dir / "model_card.json"
    required = (best, last, metrics_path, test_metrics_path, model_card_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"completed candidate lacks artifacts: {missing}")
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    if test_metrics.get("status") != "not_run":
        raise RuntimeError("independent test was accessed during teacher training")
    checkpoint = checkpoint_metadata(
        last,
        expected_architecture=matrix["common"]["architecture"],
        expected_input_size=int(matrix["common"]["input_size"]),
        expected_selection_head=candidate["specialist"],
        expected_epochs=int(matrix["common"]["epochs"]),
    )
    log_rows = training_rows(output_dir / "training_log.jsonl")
    epochs = [int(row.get("epoch", -1)) for row in log_rows]
    if epochs != list(range(len(epochs))) or not epochs:
        raise RuntimeError("candidate training epochs are not unique and contiguous")
    if epochs[-1] != checkpoint["epoch"]:
        raise RuntimeError("candidate training log and last checkpoint disagree")
    return {
        "candidate_id": candidate["candidate_id"],
        "specialist": candidate["specialist"],
        "epochs_completed": len(epochs),
        "last_epoch": epochs[-1],
        "best_checkpoint": str(best.resolve()),
        "best_checkpoint_sha256": sha256(best),
        "last_checkpoint": checkpoint,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "test_metrics": str(test_metrics_path.resolve()),
        "test_metrics_sha256": sha256(test_metrics_path),
        "model_card": str(model_card_path.resolve()),
        "model_card_sha256": sha256(model_card_path),
        "training_log_sha256": sha256(output_dir / "training_log.jsonl"),
        "test_accessed": False,
        "frozen_video_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--original-runner", type=Path, required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-last-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    for path in (args.matrix, args.state, args.original_runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.state).lower() != args.expected_state_sha256.lower():
        raise RuntimeError("interrupted Stage71 state SHA256 changed")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    validate_locked_flags(state)
    if state.get("status") != "running" or state.get("completed") or state.get("failed"):
        raise RuntimeError("Stage71 state is not the expected single-interruption state")
    if sha256(args.matrix).lower() != str(state.get("matrix_sha256", "")).lower():
        raise RuntimeError("Stage71 matrix no longer matches the interrupted state")
    if sha256(args.original_runner).lower() != str(matrix.get("runner_sha256", "")).lower():
        raise RuntimeError("original Stage71 runner no longer matches the matrix")
    if not args.output_root.is_dir():
        raise RuntimeError("interrupted Stage71 output root is missing")

    module = load_original_runner(args.original_runner)
    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"Stage71 immutable input mismatch: {name}")
        inputs[name] = path
    candidates = list(matrix.get("candidates", []))
    if len(candidates) != 2 or [item.get("specialist") for item in candidates] != ["color", "body"]:
        raise RuntimeError("unexpected Stage71 recovery candidate order")

    first = candidates[0]
    first_dir = args.output_root / first["candidate_id"]
    first_last = first_dir / "last.pt"
    if sha256(first_last).lower() != args.expected_last_sha256.lower():
        raise RuntimeError("interrupted last.pt SHA256 changed")
    first_metadata = checkpoint_metadata(
        first_last,
        expected_architecture=matrix["common"]["architecture"],
        expected_input_size=int(matrix["common"]["input_size"]),
        expected_selection_head=first["specialist"],
        expected_epochs=int(matrix["common"]["epochs"]),
    )
    log_epochs = [int(row.get("epoch", -1)) for row in training_rows(first_dir / "training_log.jsonl")]
    if log_epochs != list(range(first_metadata["epoch"] + 1)):
        raise RuntimeError("interrupted training log does not match last.pt")
    first_parameters = candidate_parameters(
        matrix, first, inputs, args.device, args.code_revision, first_dir
    )
    first_original_command = original_command(module, inputs["training_script"], first_parameters)
    if state.get("active", {}).get("command") != first_original_command:
        raise RuntimeError("interrupted active command differs from the SHA-pinned matrix")
    recovery_command = replace_init_with_resume(first_original_command, first_last)

    preflight = {
        "status": "pass_recovery_preflight",
        "checked_at": now(),
        "state_sha256": sha256(args.state),
        "matrix_sha256": sha256(args.matrix),
        "original_runner_sha256": sha256(args.original_runner),
        "recovery_script_sha256": sha256(Path(__file__).resolve()),
        "interrupted_checkpoint": first_metadata,
        "interrupted_training_epochs": log_epochs,
        "resume_command": recovery_command,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0

    recovery_started = now()
    state["recovery"] = {
        **preflight,
        "status": "running",
        "started_at": recovery_started,
        "prior_active": state.get("active"),
    }
    state["updated_at"] = now()
    atomic_json(args.state, state)
    try:
        for index, candidate in enumerate(candidates):
            candidate_id = candidate["candidate_id"]
            output_dir = args.output_root / candidate_id
            parameters = candidate_parameters(
                matrix, candidate, inputs, args.device, args.code_revision, output_dir
            )
            command = original_command(module, inputs["training_script"], parameters)
            resumed_from = None
            if index == 0:
                command = recovery_command
                resumed_from = first_metadata
            elif output_dir.exists():
                raise RuntimeError(f"remaining candidate output already exists: {output_dir}")
            if "--skip-test" not in command:
                raise RuntimeError(f"{candidate_id} lost --skip-test")
            active = {
                "candidate_id": candidate_id,
                "specialist": candidate["specialist"],
                "started_at": now(),
                "command": command,
                "recovered": index == 0,
                "resumed_from": resumed_from,
            }
            state["active"] = active
            state["updated_at"] = now()
            atomic_json(args.state, state)
            stdout_log = args.output_root / f"{candidate_id}.recovery.stdout.log"
            with stdout_log.open("ab") as handle:
                result = subprocess.run(
                    command,
                    cwd=str(inputs["training_script"].resolve().parents[1]),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"{candidate_id} returned {result.returncode}")
            verified = verify_completed_candidate(output_dir, candidate, matrix)
            state["completed"].append(
                {
                    **active,
                    **verified,
                    "finished_at": now(),
                    "return_code": result.returncode,
                    "stdout_log": str(stdout_log.resolve()),
                    "stdout_log_sha256": sha256(stdout_log),
                }
            )
            state.pop("active", None)
            state["updated_at"] = now()
            atomic_json(args.state, state)
        state["status"] = "complete_validation_only"
        state["recovery"]["status"] = "complete"
        state["recovery"]["finished_at"] = now()
        state["updated_at"] = now()
        atomic_json(args.state, state)
        evidence_path = args.output_root / "stage71-recovery-evidence.json"
        evidence = {
            **state["recovery"],
            "final_state": str(args.state.resolve()),
            "completed": state["completed"],
            "locked_flags": {key: state[key] for key in LOCKED_FALSE_FLAGS},
            "deployment_paused_by_user": True,
        }
        atomic_json(evidence_path, evidence)
        print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "evidence": str(evidence_path)}, ensure_ascii=False))
        return 0
    except Exception as error:
        state["status"] = "failed_closed_recovery"
        state.setdefault("failed", []).append(
            {
                "candidate_id": state.get("active", {}).get("candidate_id"),
                "failed_at": now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        state["recovery"]["status"] = "failed_closed"
        state["recovery"]["finished_at"] = now()
        state["updated_at"] = now()
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
