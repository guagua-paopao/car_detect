#!/usr/bin/env python3
"""Wait for Stage82 and materialize the fail-closed Stage83 v2 body manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(path: Path, **values: object) -> None:
    payload = {
        "schema_version": "stage83-body-v2-merge-state-v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_stage82_report(report_path: Path, expected_manifest: Path) -> tuple[dict, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("Stage82 did not pass")
    policy = report.get("policy", {})
    if policy.get("official_test_image_payloads_read") != 0:
        raise RuntimeError("Stage82 official test payload policy failed")
    if policy.get("validation_rows_created") != 0 or policy.get("test_rows_created") != 0:
        raise RuntimeError("Stage82 created held-out rows")
    output = report.get("output", {})
    manifest = Path(str(output.get("manifest", ""))).resolve()
    if manifest != expected_manifest.resolve():
        raise RuntimeError("Stage82 manifest path mismatch")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    actual = sha256(manifest)
    if actual.lower() != str(output.get("manifest_sha256", "")).lower():
        raise RuntimeError("Stage82 manifest SHA256 does not match its report")
    return report, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage82-report", type=Path, required=True)
    parser.add_argument("--stage82-manifest", type=Path, required=True)
    parser.add_argument("--dependency-session", default="VCAS-STAGE82-MIO-MANIFEST")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--merger", type=Path, required=True)
    parser.add_argument("--expected-merger-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()

    for path, expected, name in (
        (args.base_manifest, args.expected_base_sha256, "base"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.merger, args.expected_merger_sha256, "merger"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path).lower() != expected.lower():
            raise RuntimeError(f"{name} SHA256 mismatch")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage83 output")

    started = time.monotonic()
    write_state(args.state, status="waiting_for_stage82")
    while not args.stage82_report.is_file():
        if time.monotonic() - started > args.timeout_seconds:
            write_state(args.state, status="failed_closed", reason="Stage82 wait timeout")
            raise TimeoutError("Stage82 wait timeout")
        session = subprocess.run(
            ["tmux", "has-session", "-t", args.dependency_session],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if session.returncode != 0:
            write_state(args.state, status="failed_closed", reason="Stage82 ended without a report")
            raise RuntimeError("Stage82 ended without a report")
        time.sleep(args.poll_seconds)

    try:
        _, addition_sha = validate_stage82_report(args.stage82_report, args.stage82_manifest)
        addition_report_sha = sha256(args.stage82_report)
        write_state(
            args.state, status="running", stage82_manifest_sha256=addition_sha,
            stage82_report_sha256=addition_report_sha,
        )
        command = [
            sys.executable, str(args.merger),
            "--base-manifest", str(args.base_manifest),
            "--expected-base-sha256", args.expected_base_sha256,
            "--addition-manifest", str(args.stage82_manifest),
            "--expected-addition-sha256", addition_sha,
            "--addition-report", str(args.stage82_report),
            "--expected-addition-report-sha256", addition_report_sha,
            "--labels", str(args.labels),
            "--expected-labels-sha256", args.expected_labels_sha256,
            "--near-duplicate-hamming", "4",
            "--output-manifest", str(args.output_manifest),
            "--output-report", str(args.output_report),
        ]
        subprocess.run(command, check=True)
        write_state(
            args.state, status="complete_data_only",
            stage82_manifest_sha256=addition_sha,
            stage82_report_sha256=addition_report_sha,
            output_manifest=str(args.output_manifest.resolve()),
            output_manifest_sha256=sha256(args.output_manifest),
            output_report=str(args.output_report.resolve()),
            output_report_sha256=sha256(args.output_report),
        )
    except Exception as exc:
        write_state(args.state, status="failed_closed", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
