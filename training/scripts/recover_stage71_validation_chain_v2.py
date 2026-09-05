#!/usr/bin/env python3
"""Run the recovered Stage71 chain with fail-closed v2 before test access.

The SHA-pinned v1 recovery implementation is reused for its archive, state,
and launcher safety contracts. This wrapper inserts one validation-only stage,
requires its narrowed pair report to contain a passing pair, and makes the
one-time final-test launcher consume that v2 state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


BASE = Path("/root/autodl-tmp/vcas")
RUNS = BASE / "runs/attributes"
SCRIPTS = BASE / "code/training/scripts"
V1 = SCRIPTS / "recover_stage71_validation_chain.py"
EXPECTED_V1_SHA256 = "62d8c681a394c6b488e9bff2bf3919311844798a011b2ba580e8f45ecb278e20"
OUTPUT_ROOT = RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V2"
STATE = RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V2.state.json"
ARCHIVE_ROOT = RUNS / "ATTR-STAGE71-STALE-WAITERS-ARCHIVE-20260830-V2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_v1():
    if not V1.is_file() or sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("Stage71 v1 recovery-chain source SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("stage71_recovery_v1", V1)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load Stage71 v1 recovery chain")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patched_launchers(module) -> tuple:
    original = {name: (path, digest) for name, path, digest in module.LAUNCHERS}
    return (
        ("teacher_validation", *original["teacher_validation"]),
        ("student_training", *original["student_training"]),
        ("student_validation", *original["student_validation"]),
        (
            "fail_closed_track",
            SCRIPTS / "run_stage71_fail_closed_pair_gate_after_validation.sh",
            "7993115e7e4c2f7793687101b991fd0cdd3bc91795ca366b902d84d16d342824",
        ),
        (
            "final_test",
            SCRIPTS / "run_stage71_final_test_after_fail_closed_v2.sh",
            "c3f6606990c7390575841c9db42ae9094cc6b2b3326fa6e63d4ea1c63e6bddf4",
        ),
        ("onnx_backend", *original["onnx_backend"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()

    module = load_v1()
    module.LAUNCHERS = patched_launchers(module)
    original_run_launcher = module.run_launcher

    def run_launcher_with_v2_gate(name, launcher, expected_sha, output_root):
        record = original_run_launcher(name, launcher, expected_sha, output_root)
        if name == "fail_closed_track":
            record["gate"] = module.require_pair_gate(
                name,
                RUNS / "ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2.state.json",
                RUNS / "ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2/pair-validation-fail-closed-v2-report.json",
            )
        return record

    module.run_launcher = run_launcher_with_v2_gate
    forwarded = [
        str(V1),
        "--training-state",
        str(RUNS / "ATTR-STAGE71-CONVNEXT-TEACHERS-V1.state.json"),
        "--training-session",
        "VCAS-RECOVER-STAGE71-TEACHERS",
        "--output-root",
        str(OUTPUT_ROOT),
        "--state",
        str(STATE),
        "--archive-root",
        str(ARCHIVE_ROOT),
        "--timeout-hours",
        str(args.timeout_hours),
    ]
    if args.preflight_only:
        forwarded.append("--preflight-only")
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        result = module.main()
    finally:
        sys.argv = original_argv

    if args.preflight_only:
        return result
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state["schema_version"] = "stage71-recovered-validation-chain-v2"
        state["v1_recovery_source"] = str(V1)
        state["v1_recovery_source_sha256"] = EXPECTED_V1_SHA256
        state["fail_closed_v2_required_before_test"] = True
        state["wrapper_sha256"] = sha256(Path(__file__).resolve())
        atomic_json(STATE, state)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
