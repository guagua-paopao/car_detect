#!/usr/bin/env python3
"""Continue the Stage71 validation/distillation chain after host interruption.

Only empty, SHA-pinned waiting artifacts are archived. Each downstream stage is
started synchronously and only after the previous stage produced its expected
fail-closed gate. The frozen video and deployment are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE = Path("/root/autodl-tmp/vcas")
RUNS = BASE / "runs/attributes"
SCRIPTS = BASE / "code/training/scripts"

LAUNCHERS = (
    (
        "teacher_validation",
        SCRIPTS / "run_stage71_teacher_pair_validation_only.sh",
        "423919fd1e2026060ead74563b3ac43219b9955eb28d2613dcd9dd4fd5035977",
    ),
    (
        "student_training",
        SCRIPTS / "run_stage71_students_after_teacher_validation_v2.sh",
        "0859b72549be4ccc5ce5032db5c2df827828fc055f4673db74edfd0b2e2a9cd4",
    ),
    (
        "student_validation",
        SCRIPTS / "run_stage71_student_pair_validation_after_training.sh",
        "6902fce2aba31d4ca7ff315abb7c91ab975cd10752067d78e4f5509dd01284bf",
    ),
    (
        "final_test",
        SCRIPTS / "run_stage71_final_test_after_student_validation.sh",
        "2df87d84f06d489fe6623785d374e083529cf911dc4de5889a64f04e77c8a487",
    ),
    (
        "onnx_backend",
        SCRIPTS / "run_stage71_onnx_after_final_test_v2.sh",
        "0febb7153be3b255be597c01b8fe693572fe678914cc8db7c1d462c41ba6d928",
    ),
)

STALE_WAITERS = {
    "teacher_validation": {
        "directory": RUNS / "ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1",
        "state": RUNS / "ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1.state.json",
        "state_sha256": "95ff86bdb845bbf980370d5766a99505e7609f954f492a71cf39efc4b9811ac8",
        "status": "waiting_for_training",
    },
    "final_test": {
        "directory": RUNS / "ATTR-STAGE71-FINAL-TEST-V1",
        "state": RUNS / "ATTR-STAGE71-FINAL-TEST-V1.state.json",
        "state_sha256": "f1a9622916968003d5e022f854f8bda136220508c9df553ff2708100f6a47e77",
        "status": "waiting_for_student_validation",
    },
    "onnx_backend": {
        "directory": RUNS / "ATTR-STAGE71-BACKEND-CANDIDATE-V2",
        "state": RUNS / "ATTR-STAGE71-BACKEND-CANDIDATE-V2.state.json",
        "state_sha256": "ed96f45517ada809ca9e6e3067e50a17b42f0232cf96cb6b69d0b11512aa4078",
        "status": "waiting_for_final_test",
    },
}

LOCKED_FALSE_FLAGS = (
    "frozen_video_used",
    "production_model_modified",
    "deployment_performed",
)


class GateClosed(RuntimeError):
    def __init__(self, stage: str, status: str):
        super().__init__(f"{stage} stopped fail closed with status={status}")
        self.stage = stage
        self.status = status


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def within(path: Path, parent: Path) -> Path:
    resolved = path.resolve(strict=path.exists())
    resolved.relative_to(parent.resolve(strict=True))
    return resolved


def validate_stale_waiter(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    directory = spec["directory"]
    state_path = spec["state"]
    within(directory, RUNS)
    within(state_path, RUNS)
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"{name} stale directory is missing or linked")
    children = list(directory.iterdir())
    if children:
        raise RuntimeError(f"{name} stale directory is not empty: {children}")
    if not state_path.is_file() or state_path.is_symlink():
        raise RuntimeError(f"{name} stale state is missing or linked")
    actual_sha = sha256(state_path)
    if actual_sha != spec["state_sha256"]:
        raise RuntimeError(f"{name} stale state SHA256 changed")
    state = read_json(state_path)
    if state.get("status") != spec["status"]:
        raise RuntimeError(f"{name} stale state status changed")
    for key in LOCKED_FALSE_FLAGS:
        if state.get(key) is not False:
            raise RuntimeError(f"{name} stale state violates {key}")
    if state.get("deployment_paused_by_user") is not True:
        raise RuntimeError(f"{name} stale state lost deployment hold")
    return {
        "name": name,
        "directory": str(directory),
        "directory_empty": True,
        "state": str(state_path),
        "state_sha256": actual_sha,
        "status": state["status"],
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }


def archive_stale_waiter(
    name: str,
    spec: dict[str, Any],
    archive_root: Path,
    archive_records: list[dict[str, Any]],
) -> None:
    record = validate_stale_waiter(name, spec)
    archive_root_resolved = archive_root.resolve(strict=False)
    archive_root_resolved.relative_to(RUNS.resolve(strict=True))
    archive_root.mkdir(parents=False, exist_ok=True)
    targets = {
        spec["directory"]: archive_root / spec["directory"].name,
        spec["state"]: archive_root / spec["state"].name,
    }
    for source, target in targets.items():
        within(source, RUNS)
        target.resolve(strict=False).relative_to(archive_root.resolve(strict=True))
        if target.exists():
            raise FileExistsError(f"archive target already exists: {target}")
    for source, target in targets.items():
        source.rename(target)
    record.update(
        {
            "archived_at": now(),
            "archived_directory": str(targets[spec["directory"]]),
            "archived_state": str(targets[spec["state"]]),
        }
    )
    archive_records.append(record)
    atomic_json(
        archive_root / "archive-manifest.json",
        {
            "schema_version": "stage71-stale-waiter-archive-v1",
            "created_at": archive_records[0]["archived_at"],
            "updated_at": now(),
            "records": archive_records,
            "recoverable": True,
            "original_data_deleted": False,
        },
    )


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_teacher_recovery(
    training_state: Path, session: str, timeout_hours: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if training_state.is_file():
            state = read_json(training_state)
            status = state.get("status")
            if status == "complete_validation_only":
                if len(state.get("completed", [])) != 2 or state.get("failed"):
                    raise RuntimeError("teacher recovery completion evidence is incomplete")
                for key in LOCKED_FALSE_FLAGS:
                    if state.get(key) is not False:
                        raise RuntimeError(f"teacher recovery violates {key}")
                return state
            if status not in {"running"}:
                raise RuntimeError(f"teacher recovery failed with status={status}")
        if not tmux_alive(session):
            raise RuntimeError("teacher recovery session ended without completion evidence")
        time.sleep(30)
    raise TimeoutError("timed out waiting for teacher recovery")


def run_launcher(
    name: str, launcher: Path, expected_sha: str, output_root: Path
) -> dict[str, Any]:
    if not launcher.is_file() or sha256(launcher) != expected_sha:
        raise RuntimeError(f"{name} launcher SHA256 mismatch")
    log = output_root / f"{name}.log"
    started_at = now()
    with log.open("wb") as handle:
        result = subprocess.run(
            ["bash", str(launcher)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "stage": name,
        "launcher": str(launcher),
        "launcher_sha256": expected_sha,
        "started_at": started_at,
        "finished_at": now(),
        "return_code": result.returncode,
        "log": str(log),
        "log_sha256": sha256(log),
    }
    return record


def require_pair_gate(
    stage: str, state_path: Path, report_path: Path
) -> dict[str, Any]:
    state = read_json(state_path)
    if state.get("status") != "complete":
        raise GateClosed(stage, str(state.get("status")))
    report = read_json(report_path)
    status = str(report.get("status"))
    passing = list(report.get("passing_pairs", []))
    if status != "pass_pairs_available" or not passing:
        raise GateClosed(stage, status)
    for key in LOCKED_FALSE_FLAGS:
        if report.get("policy", {}).get(key) is not False:
            raise RuntimeError(f"{stage} report violates {key}")
    return {
        "state": str(state_path),
        "state_sha256": sha256(state_path),
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "passing_pairs": passing,
    }


def require_state_status(
    stage: str, state_path: Path, accepted: set[str]
) -> dict[str, Any]:
    state = read_json(state_path)
    status = str(state.get("status"))
    if status not in accepted:
        raise GateClosed(stage, status)
    for key in LOCKED_FALSE_FLAGS:
        if state.get(key) is not False:
            raise RuntimeError(f"{stage} state violates {key}")
    return {
        "state": str(state_path),
        "state_sha256": sha256(state_path),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-state",
        type=Path,
        default=RUNS / "ATTR-STAGE71-CONVNEXT-TEACHERS-V1.state.json",
    )
    parser.add_argument(
        "--training-session", default="VCAS-RECOVER-STAGE71-TEACHERS"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V1",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V1.state.json",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=RUNS / "ATTR-STAGE71-STALE-WAITERS-ARCHIVE-20260829T1537Z",
    )
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    for path in (args.output_root, args.state, args.archive_root):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage71 recovery-chain evidence: {path}")
        path.resolve(strict=False).relative_to(RUNS.resolve(strict=True))
    for _, launcher, expected_sha in LAUNCHERS:
        if not launcher.is_file() or sha256(launcher) != expected_sha:
            raise RuntimeError(f"launcher preflight failed: {launcher}")
    stale_preflight = {
        name: validate_stale_waiter(name, spec)
        for name, spec in STALE_WAITERS.items()
    }
    preflight = {
        "status": "pass_recovery_chain_preflight",
        "checked_at": now(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "launchers": [
            {
                "stage": stage,
                "path": str(launcher),
                "sha256": expected_sha,
            }
            for stage, launcher, expected_sha in LAUNCHERS
        ],
        "stale_waiters": stale_preflight,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=False, exist_ok=False)
    chain = {
        "schema_version": "stage71-recovered-validation-chain-v1",
        "status": "waiting_for_teacher_recovery",
        "created_at": now(),
        "updated_at": now(),
        "training_state": str(args.training_state),
        "training_session": args.training_session,
        "preflight": preflight,
        "completed_stages": [],
        "archived_waiters": [],
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, chain)
    try:
        training = wait_for_teacher_recovery(
            args.training_state, args.training_session, args.timeout_hours
        )
        chain["teacher_training_state_sha256"] = sha256(args.training_state)
        chain["teacher_candidates"] = [
            item.get("candidate_id") for item in training.get("completed", [])
        ]

        archive_stale_waiter(
            "teacher_validation",
            STALE_WAITERS["teacher_validation"],
            args.archive_root,
            chain["archived_waiters"],
        )
        for stage, launcher, expected_sha in LAUNCHERS:
            if stage == "final_test":
                archive_stale_waiter(
                    stage,
                    STALE_WAITERS[stage],
                    args.archive_root,
                    chain["archived_waiters"],
                )
                chain["test_accessed"] = True
            elif stage == "onnx_backend":
                archive_stale_waiter(
                    stage,
                    STALE_WAITERS[stage],
                    args.archive_root,
                    chain["archived_waiters"],
                )
                chain["backend_gates_run"] = True
            chain["status"] = f"running_{stage}"
            chain["updated_at"] = now()
            atomic_json(args.state, chain)
            record = run_launcher(stage, launcher, expected_sha, args.output_root)
            chain["last_stage_record"] = record
            chain["updated_at"] = now()
            atomic_json(args.state, chain)
            if stage == "teacher_validation":
                record["gate"] = require_pair_gate(
                    stage,
                    RUNS / "ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1.state.json",
                    RUNS / "ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1/pair-validation-screen-report.json",
                )
            elif stage == "student_training":
                record["gate"] = require_state_status(
                    stage,
                    RUNS / "ATTR-STAGE71-STUDENTS-V1.state.json",
                    {"complete_validation_only"},
                )
            elif stage == "student_validation":
                record["gate"] = require_pair_gate(
                    stage,
                    RUNS / "ATTR-STAGE71-STUDENT-PAIR-VALIDATION-ONLY-V1.state.json",
                    RUNS / "ATTR-STAGE71-STUDENT-PAIR-VALIDATION-ONLY-V1/pair-validation-screen-report.json",
                )
            elif stage == "final_test":
                record["gate"] = require_state_status(
                    stage,
                    RUNS / "ATTR-STAGE71-FINAL-TEST-V1.state.json",
                    {"pass_backend_eligible"},
                )
            elif stage == "onnx_backend":
                record["gate"] = require_state_status(
                    stage,
                    RUNS / "ATTR-STAGE71-BACKEND-CANDIDATE-V2.state.json",
                    {"pass_onnx_waiting_local_tensorrt_cpp"},
                )
            if record["return_code"] != 0:
                raise RuntimeError(
                    f"{stage} launcher returned {record['return_code']}; see {record['log']}"
                )
            chain["completed_stages"].append(record)
            chain["updated_at"] = now()
            atomic_json(args.state, chain)
        chain["status"] = "pass_onnx_waiting_local_tensorrt_cpp"
        chain["updated_at"] = now()
        atomic_json(args.state, chain)
        return 0
    except GateClosed as error:
        chain["status"] = f"stopped_fail_closed_{error.stage}"
        chain["gate_status"] = error.status
        chain["updated_at"] = now()
        atomic_json(args.state, chain)
        return 0
    except Exception as error:
        chain["status"] = "failed_closed_recovery_chain"
        chain["error"] = repr(error)
        chain["traceback"] = traceback.format_exc()
        chain["updated_at"] = now()
        atomic_json(args.state, chain)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
