#!/usr/bin/env python3
"""Retry a guarded Google Drive download while preserving resumable partials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def retry_delay(attempt: int, initial_seconds: float, maximum_seconds: float) -> float:
    return min(maximum_seconds, initial_seconds * (2 ** max(0, attempt - 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--retry-state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gdown-pythonpath", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--min-free-bytes", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--initial-delay-seconds", type=float, default=30.0)
    parser.add_argument("--max-delay-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("attempts must be positive")
    retry_state = {
        "schema_version": "guarded-gdrive-retry-v1",
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "attempts_planned": args.attempts,
        "attempts_completed": 0,
        "output": str(args.output),
        "download_state": str(args.download_state),
    }
    write_json(args.retry_state, retry_state)

    command = [
        str(args.python),
        str(args.guard_script),
        "--file-id", args.file_id,
        "--output", str(args.output),
        "--allowed-root", str(args.allowed_root),
        "--state", str(args.download_state),
        "--log", str(args.log),
        "--gdown-pythonpath", str(args.gdown_pythonpath),
        "--max-bytes", str(args.max_bytes),
        "--min-free-bytes", str(args.min_free_bytes),
        "--poll-seconds", str(args.poll_seconds),
    ]
    for attempt in range(1, args.attempts + 1):
        retry_state.update(current_attempt=attempt, updated_at=utc_now())
        write_json(args.retry_state, retry_state)
        result = subprocess.run(command, check=False)
        download_state = read_json(args.download_state)
        retry_state.update(
            attempts_completed=attempt,
            last_return_code=result.returncode,
            last_download_status=download_state.get("status"),
            effective_download_bytes=download_state.get("effective_download_bytes"),
            updated_at=utc_now(),
        )
        if result.returncode == 0 and download_state.get("status") == "complete_pending_archive_audit":
            retry_state.update(status="complete_pending_archive_audit")
            write_json(args.retry_state, retry_state)
            return 0
        if download_state.get("status") == "stopped_by_resource_guard":
            retry_state.update(status="stopped_by_resource_guard")
            write_json(args.retry_state, retry_state)
            return 2
        if attempt < args.attempts:
            delay = retry_delay(attempt, args.initial_delay_seconds, args.max_delay_seconds)
            retry_state.update(status="waiting_to_retry", next_delay_seconds=delay, updated_at=utc_now())
            write_json(args.retry_state, retry_state)
            time.sleep(delay)
            retry_state.update(status="running", updated_at=utc_now())
            write_json(args.retry_state, retry_state)

    retry_state.update(status="retries_exhausted_preserved", updated_at=utc_now())
    write_json(args.retry_state, retry_state)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
