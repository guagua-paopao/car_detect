#!/usr/bin/env python3
"""Resume only an interrupted Stage71 body teacher after color completed."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


def load_common(path: Path):
    spec = importlib.util.spec_from_file_location("stage71_recovery_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load common recovery module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_interrupted_body_state(state: dict[str, Any]) -> None:
    if state.get("status") != "running" or state.get("failed"):
        raise RuntimeError("Stage71 is not a clean running state")
    completed = state.get("completed", [])
    if len(completed) != 1:
        raise RuntimeError("body recovery requires exactly one completed teacher")
    if completed[0].get("specialist") != "color":
        raise RuntimeError("the completed Stage71 teacher is not color")
    active = state.get("active", {})
    if active.get("specialist") != "body":
        raise RuntimeError("the interrupted Stage71 teacher is not body")


def require_completed_record_matches(record: dict[str, Any], verified: dict[str, Any]) -> None:
    keys = (
        "candidate_id", "specialist", "last_epoch", "best_checkpoint_sha256",
        "metrics_sha256", "test_metrics_sha256", "model_card_sha256",
        "training_log_sha256",
    )
    for key in keys:
        if record.get(key) != verified.get(key):
            raise RuntimeError(f"completed color evidence changed: {key}")
    recorded_last = record.get("last_checkpoint", {})
    verified_last = verified.get("last_checkpoint", {})
    for key in ("sha256", "epoch", "best_score"):
        if recorded_last.get(key) != verified_last.get(key):
            raise RuntimeError(f"completed color last checkpoint changed: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--original-runner", type=Path, required=True)
    parser.add_argument("--common-recovery", type=Path, required=True)
    parser.add_argument("--expected-common-recovery-sha256", required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-body-last-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = load_common(args.common_recovery)
    for path in (args.matrix, args.state, args.original_runner, args.common_recovery):
        if not path.is_file():
            raise FileNotFoundError(path)
    if common.sha256(args.common_recovery).lower() != args.expected_common_recovery_sha256.lower():
        raise RuntimeError("common recovery implementation SHA256 changed")
    if common.sha256(args.state).lower() != args.expected_state_sha256.lower():
        raise RuntimeError("interrupted Stage71 body state SHA256 changed")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    common.validate_locked_flags(state)
    validate_interrupted_body_state(state)
    if common.sha256(args.matrix).lower() != str(state.get("matrix_sha256", "")).lower():
        raise RuntimeError("Stage71 matrix no longer matches body state")
    if common.sha256(args.original_runner).lower() != str(matrix.get("runner_sha256", "")).lower():
        raise RuntimeError("original Stage71 runner no longer matches matrix")

    module = common.load_original_runner(args.original_runner)
    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or common.sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"Stage71 immutable input mismatch: {name}")
        inputs[name] = path
    candidates = list(matrix.get("candidates", []))
    if len(candidates) != 2 or [item.get("specialist") for item in candidates] != ["color", "body"]:
        raise RuntimeError("unexpected Stage71 teacher order")
    color, body = candidates

    color_dir = args.output_root / color["candidate_id"]
    verified_color = common.verify_completed_candidate(color_dir, color, matrix)
    require_completed_record_matches(state["completed"][0], verified_color)

    body_dir = args.output_root / body["candidate_id"]
    body_last = body_dir / "last.pt"
    if common.sha256(body_last).lower() != args.expected_body_last_sha256.lower():
        raise RuntimeError("interrupted body last.pt SHA256 changed")
    body_metadata = common.checkpoint_metadata(
        body_last,
        expected_architecture=matrix["common"]["architecture"],
        expected_input_size=int(matrix["common"]["input_size"]),
        expected_selection_head="body",
        expected_epochs=int(matrix["common"]["epochs"]),
    )
    log_epochs = [
        int(row.get("epoch", -1))
        for row in common.training_rows(body_dir / "training_log.jsonl")
    ]
    if log_epochs != list(range(body_metadata["epoch"] + 1)):
        raise RuntimeError("body training log does not match last.pt")
    parameters = common.candidate_parameters(
        matrix, body, inputs, args.device, args.code_revision, body_dir
    )
    original = common.original_command(module, inputs["training_script"], parameters)
    if state.get("active", {}).get("command") != original:
        raise RuntimeError("interrupted body command differs from the pinned matrix")
    command = common.replace_init_with_resume(original, body_last)
    preflight = {
        "status": "pass_body_recovery_preflight",
        "checked_at": common.now(),
        "state_sha256": common.sha256(args.state),
        "matrix_sha256": common.sha256(args.matrix),
        "original_runner_sha256": common.sha256(args.original_runner),
        "common_recovery_sha256": common.sha256(args.common_recovery),
        "body_recovery_script_sha256": common.sha256(Path(__file__).resolve()),
        "completed_color_checkpoint_sha256": verified_color["best_checkpoint_sha256"],
        "interrupted_body_checkpoint": body_metadata,
        "interrupted_body_epochs": log_epochs,
        "resume_command": command,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0

    state["body_recovery"] = {**preflight, "status": "running", "started_at": common.now()}
    state["active"] = {
        "candidate_id": body["candidate_id"],
        "specialist": "body",
        "started_at": common.now(),
        "command": command,
        "recovered": True,
        "resumed_from": body_metadata,
    }
    state["updated_at"] = common.now()
    common.atomic_json(args.state, state)
    stdout_log = args.output_root / f"{body['candidate_id']}.body-recovery.stdout.log"
    try:
        with stdout_log.open("ab") as handle:
            result = subprocess.run(
                command,
                cwd=str(inputs["training_script"].resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"body teacher returned {result.returncode}")
        verified_body = common.verify_completed_candidate(body_dir, body, matrix)
        state["completed"].append({
            **state["active"], **verified_body, "finished_at": common.now(),
            "return_code": 0, "stdout_log": str(stdout_log.resolve()),
            "stdout_log_sha256": common.sha256(stdout_log),
        })
        state.pop("active", None)
        state["status"] = "complete_validation_only"
        state["body_recovery"]["status"] = "complete"
        state["body_recovery"]["finished_at"] = common.now()
        state["updated_at"] = common.now()
        common.atomic_json(args.state, state)
        evidence = args.output_root / "stage71-body-recovery-evidence.json"
        common.atomic_json(evidence, {
            **state["body_recovery"], "final_state": str(args.state.resolve()),
            "completed": state["completed"],
        })
        print(json.dumps({"status": state["status"], "evidence": str(evidence)}, ensure_ascii=False))
        return 0
    except Exception as error:
        state["status"] = "failed_closed_body_recovery"
        state.setdefault("failed", []).append({
            "candidate_id": body["candidate_id"], "failed_at": common.now(),
            "error": repr(error), "traceback": traceback.format_exc(),
        })
        state["body_recovery"]["status"] = "failed_closed"
        state["body_recovery"]["finished_at"] = common.now()
        state["updated_at"] = common.now()
        common.atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    sys.exit(main())
