#!/usr/bin/env python3
"""Stage71 recovered chain with fixed fail-closed v2 test gating.

This successor reuses the SHA-pinned v2 chain and replaces only the final-test
launcher. The replacement first runs the one validation-selected pair once on
test, then applies the same fixed fail-closed temporal parameters before ONNX
can be unlocked.
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
V2 = SCRIPTS / "recover_stage71_validation_chain_v2.py"
EXPECTED_V2_SHA256 = "0ff1f89982f08a33cfafff66495100b63282d57c7bdc4db430ce2196bdb0aa22"
OUTPUT_ROOT = RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V3"
STATE = RUNS / "ATTR-STAGE71-RECOVERED-CHAIN-V3.state.json"
ARCHIVE_ROOT = RUNS / "ATTR-STAGE71-STALE-WAITERS-ARCHIVE-20260830-V3"


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


def load_v2():
    if not V2.is_file() or sha256(V2) != EXPECTED_V2_SHA256:
        raise RuntimeError("Stage71 v2 recovery-chain source SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("stage71_recovery_v2", V2)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load Stage71 v2 recovery chain")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_final_launcher(launchers: tuple) -> tuple:
    result = []
    found = False
    for stage, path, digest in launchers:
        if stage == "final_test":
            result.append(
                (
                    "final_test",
                    SCRIPTS / "run_stage71_final_test_after_fail_closed_v3.sh",
                    "bdef986408150e0e490507db4b39d7fc769814c2b7ae9cb92ccaf70f55d39387",
                )
            )
            found = True
        else:
            result.append((stage, path, digest))
    if not found:
        raise RuntimeError("v2 chain has no final-test stage")
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()

    module = load_v2()
    original_patched_launchers = module.patched_launchers

    def patched_launchers_v3(base_module):
        return replace_final_launcher(original_patched_launchers(base_module))

    module.patched_launchers = patched_launchers_v3
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.STATE = STATE
    module.ARCHIVE_ROOT = ARCHIVE_ROOT
    forwarded = [str(V2), "--timeout-hours", str(args.timeout_hours)]
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
        state["schema_version"] = "stage71-recovered-validation-chain-v3"
        state["v2_recovery_source"] = str(V2)
        state["v2_recovery_source_sha256"] = EXPECTED_V2_SHA256
        state["fixed_fail_closed_v2_test_required_before_onnx"] = True
        state["v3_wrapper_sha256"] = sha256(Path(__file__).resolve())
        atomic_json(STATE, state)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
