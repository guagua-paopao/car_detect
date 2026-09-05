#!/usr/bin/env python3
"""Run Stage108 three-branch ONNX export and validation-only parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(command: list[str], log: Path) -> int:
    with log.open("wb") as handle:
        return subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode


def validate_stage107_state(
    state_path: Path, expected_sha256: str
) -> tuple[Path, str]:
    require_sha(state_path, expected_sha256, "Stage107 state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "pass_backend_eligible" or state.get("backend_eligible") is not True:
        raise RuntimeError("Stage107 did not authorize backend execution")
    if state.get("test_used_for_selection") is not False or state.get("frozen_video_used") is not False:
        raise RuntimeError("Stage107 state has unsafe test/frozen policy")
    report_path = Path(str(state.get("report", "")))
    report_sha = str(state.get("report_sha256", ""))
    if not report_sha:
        raise RuntimeError("Stage107 state has no report SHA256")
    require_sha(report_path, report_sha, "Stage107 report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass_backend_eligible":
        raise RuntimeError("Stage107 report is not backend eligible")
    if not report.get("composite_gates") or not all(report["composite_gates"].values()):
        raise RuntimeError("Stage107 report has failed composite gates")
    return report_path, report_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage107-state", type=Path, required=True)
    parser.add_argument("--expected-stage107-state-sha256", required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--parity-validator", type=Path, required=True)
    parser.add_argument("--expected-parity-validator-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--expected-validation-manifest-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage108 evidence")
    pins = (
        (args.exporter, args.expected_exporter_sha256, "Stage108 exporter"),
        (args.parity_validator, args.expected_parity_validator_sha256, "Stage108 parity validator"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (
            args.validation_manifest,
            args.expected_validation_manifest_sha256,
            "validation manifest",
        ),
    )
    for path, expected, label in pins:
        require_sha(path, expected, label)
    stage107_report, stage107_report_sha = validate_stage107_state(
        args.stage107_state, args.expected_stage107_state_sha256
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "stage108-three-branch-onnx-state-v1",
        "status": "running_onnx_export",
        "created_at": now(),
        "updated_at": now(),
        "stage107_state": str(args.stage107_state.resolve()),
        "stage107_state_sha256": sha256(args.stage107_state),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "tensorrt_run": False,
        "deployment_performed": False,
    }
    atomic_json(args.state, state)
    try:
        export_dir = args.output_root / "onnx"
        export_rc = run(
            [
                args.python,
                str(args.exporter),
                "--stage107-report", str(stage107_report),
                "--expected-stage107-report-sha256", stage107_report_sha,
                "--labels", str(args.labels),
                "--expected-labels-sha256", args.expected_labels_sha256,
                "--output-dir", str(export_dir),
                "--opset", "17",
            ],
            args.output_root / "onnx-export.log",
        )
        if export_rc != 0:
            raise RuntimeError(f"Stage108 ONNX export failed with exit code {export_rc}")
        export_report = export_dir / "onnx-export-report.json"
        if not export_report.is_file():
            raise RuntimeError("Stage108 ONNX export report missing")
        state.update({"status": "running_onnx_parity", "updated_at": now()})
        atomic_json(args.state, state)
        parity_report = args.output_root / "onnx-parity.json"
        parity_rc = run(
            [
                args.python,
                str(args.parity_validator),
                "--export-report", str(export_report),
                "--expected-export-report-sha256", sha256(export_report),
                "--manifest", str(args.validation_manifest),
                "--expected-manifest-sha256", args.expected_validation_manifest_sha256,
                "--datasets-safety-root", str(args.datasets_safety_root),
                "--output", str(parity_report),
                "--limit", "1024",
                "--batch-size", "64",
                "--device", args.device,
            ],
            args.output_root / "onnx-parity.log",
        )
        if not parity_report.is_file():
            raise RuntimeError(f"Stage108 parity produced no report; exit code {parity_rc}")
        parity = json.loads(parity_report.read_text(encoding="utf-8"))
        passed = parity_rc == 0 and parity.get("status") == "pass" and parity.get("gate") is True
        state.update(
            {
                "status": "pass_onnx_parity" if passed else "fail_closed_onnx_parity",
                "updated_at": now(),
                "export_report": str(export_report.resolve()),
                "export_report_sha256": sha256(export_report),
                "parity_report": str(parity_report.resolve()),
                "parity_report_sha256": sha256(parity_report),
                "onnx_parity_pass": passed,
                "tensorrt_authorized": passed,
            }
        )
        atomic_json(args.state, state)
        print(json.dumps({"status": state["status"], "tensorrt_authorized": passed}))
        return 0
    except Exception as error:
        state.update(
            {
                "status": "failed_closed_runtime",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "tensorrt_authorized": False,
            }
        )
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
