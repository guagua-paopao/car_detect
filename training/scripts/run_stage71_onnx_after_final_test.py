#!/usr/bin/env python3
"""Wait for a passing Stage71 final test, then export and validate ONNX only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def session_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_final_test(path: Path, session: str, timeout_hours: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    waiting = {
        "waiting_for_student_validation",
        "running_single_selected_pair_test",
    }
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            status = state.get("status")
            if status == "pass_backend_eligible":
                return state
            if status not in waiting:
                raise RuntimeError(f"final test failed closed with status={status!r}")
        if not session_alive(session):
            raise RuntimeError("final-test session ended without passing evidence")
        time.sleep(30)
    raise TimeoutError("timed out waiting for Stage71 final test")


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"backend command failed ({result.returncode}); see {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test-state", type=Path, required=True)
    parser.add_argument("--final-test-session", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--expected-parity-validator-sha256", required=True)
    parser.add_argument("--expected-trt-input-preparer-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--expected-validation-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    args = parser.parse_args()

    if args.output_root.exists() or args.state.exists():
        raise RuntimeError("refusing to overwrite Stage71 ONNX backend evidence")
    exporter = args.scripts_root / "export_stage71_decoupled_onnx.py"
    parity_validator = args.scripts_root / "validate_stage71_decoupled_onnx_parity.py"
    trt_input_preparer = args.scripts_root / "prepare_stage71_attribute_trt_parity.py"
    pins = {
        "exporter": (exporter, args.expected_exporter_sha256),
        "parity validator": (
            parity_validator,
            args.expected_parity_validator_sha256,
        ),
        "TensorRT input preparer": (
            trt_input_preparer,
            args.expected_trt_input_preparer_sha256,
        ),
        "labels": (args.labels, args.expected_labels_sha256),
        "validation manifest": (
            args.validation_manifest,
            args.expected_validation_manifest_sha256,
        ),
    }
    for label, (path, expected) in pins.items():
        require_sha(path, expected, label)

    args.output_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "stage71-onnx-backend-v1",
        "status": "waiting_for_final_test",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "local_test_inference_run": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "production_registry_modified": False,
        "production_config_modified": False,
        "tensorrt_run": False,
        "cpp_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)

    try:
        final_state = wait_for_final_test(
            args.final_test_state,
            args.final_test_session,
            args.timeout_hours,
        )
        final_report = Path(str(final_state.get("report", "")))
        final_report_sha = str(final_state.get("report_sha256", ""))
        if not final_report_sha:
            raise RuntimeError("passing final-test state has no report SHA256")
        require_sha(final_report, final_report_sha, "final-test report")
        state.update(
            {
                "status": "running_onnx_export_and_parity",
                "updated_at": utc_now(),
                "upstream_test_accessed": True,
                "upstream_test_used_for_selection": False,
                "final_test_state": str(args.final_test_state.resolve()),
                "final_test_state_sha256": sha256(args.final_test_state),
                "final_test_report": str(final_report.resolve()),
                "final_test_report_sha256": final_report_sha,
            }
        )
        atomic_json(args.state, state)

        run(
            [
                args.python,
                str(exporter),
                "--final-test-report",
                str(final_report),
                "--expected-final-test-report-sha256",
                final_report_sha,
                "--labels",
                str(args.labels),
                "--expected-labels-sha256",
                args.expected_labels_sha256,
                "--output-dir",
                str(args.output_root / "onnx"),
            ],
            args.output_root / "onnx-export.log",
        )
        export_report = args.output_root / "onnx" / "onnx-export-report.json"
        export_report_sha = sha256(export_report)
        parity_report = args.output_root / "onnx-parity.json"
        run(
            [
                args.python,
                str(parity_validator),
                "--export-report",
                str(export_report),
                "--expected-export-report-sha256",
                export_report_sha,
                "--manifest",
                str(args.validation_manifest),
                "--expected-manifest-sha256",
                args.expected_validation_manifest_sha256,
                "--output",
                str(parity_report),
                "--split",
                "validation",
                "--limit",
                "1024",
                "--batch-size",
                "64",
                "--device",
                args.device,
            ],
            args.output_root / "onnx-parity.log",
        )
        parity = json.loads(parity_report.read_text(encoding="utf-8"))
        if parity.get("status") != "pass" or parity.get("gate") is not True:
            raise RuntimeError("ONNX parity did not pass")
        trt_inputs = args.output_root / "trt-parity-inputs"
        run(
            [
                args.python,
                str(trt_input_preparer),
                "--export-report",
                str(export_report),
                "--expected-export-report-sha256",
                export_report_sha,
                "--manifest",
                str(args.validation_manifest),
                "--expected-manifest-sha256",
                args.expected_validation_manifest_sha256,
                "--output-dir",
                str(trt_inputs),
                "--split",
                "validation",
                "--batch-size",
                "8",
            ],
            args.output_root / "trt-parity-inputs.log",
        )
        trt_input_manifest = trt_inputs / "manifest.json"
        trt_input_document = json.loads(trt_input_manifest.read_text(encoding="utf-8"))
        if trt_input_document.get("status") != "pass_inputs_prepared":
            raise RuntimeError("TensorRT parity inputs were not prepared")
        for label, (path, expected) in pins.items():
            require_sha(path, expected, label)
        require_sha(final_report, final_report_sha, "final-test report")
        state.update(
            {
                "status": "pass_onnx_waiting_local_tensorrt_cpp",
                "updated_at": utc_now(),
                "onnx_export_report": str(export_report.resolve()),
                "onnx_export_report_sha256": export_report_sha,
                "onnx_parity_report": str(parity_report.resolve()),
                "onnx_parity_report_sha256": sha256(parity_report),
                "onnx_parity_gate": True,
                "trt_parity_input_manifest": str(trt_input_manifest.resolve()),
                "trt_parity_input_manifest_sha256": sha256(trt_input_manifest),
                "next_action": "copy isolated candidate artifacts to the target Windows host and run TensorRT/C++/real-engine gates without deployment",
            }
        )
        atomic_json(args.state, state)
        print(json.dumps({"status": state["status"], "state": str(args.state)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        state.update(
            {
                "status": "failed_closed",
                "updated_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
