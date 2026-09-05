from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_within(path: Path, allowed_root: Path) -> Path:
    resolved = path.resolve()
    root = allowed_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"output escapes allowed root: {resolved} not under {root}") from error
    return resolved


def guard_reason(*, output_bytes: int, free_bytes: int, max_bytes: int, min_free_bytes: int) -> str | None:
    if output_bytes > max_bytes:
        return f"max_bytes_exceeded:{output_bytes}>{max_bytes}"
    if free_bytes < min_free_bytes:
        return f"min_free_bytes_breached:{free_bytes}<{min_free_bytes}"
    return None


def effective_download_bytes(*, output_bytes: int, initial_free_bytes: int, current_free_bytes: int) -> int:
    """Bound temporary/opaque downloader files by observed filesystem consumption."""
    return max(output_bytes, max(0, initial_free_bytes - current_free_bytes))


def temporary_download_bytes(output: Path) -> int:
    return sum(
        path.stat().st_size
        for path in output.parent.glob(f"{output.name}*.part")
        if path.is_file()
    )


def write_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gdown with hard archive-size and disk-free guards.")
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gdown-pythonpath", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--min-free-bytes", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.allowed_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = ensure_within(args.output, root)
    state_path = ensure_within(args.state, root)
    log_path = ensure_within(args.log, root)
    gdown_pythonpath = args.gdown_pythonpath.resolve()
    if not gdown_pythonpath.is_dir():
        raise FileNotFoundError(gdown_pythonpath)
    if args.max_bytes <= 0 or args.min_free_bytes <= 0:
        raise ValueError("guards must be positive")
    if args.poll_seconds < 0.25:
        raise ValueError("poll interval is too small")

    initial_free = shutil.disk_usage(root).free
    initial_output_size = output.stat().st_size if output.exists() else 0
    initial_temporary_size = temporary_download_bytes(output)
    initial_size = max(initial_output_size, initial_temporary_size)
    initial_reason = guard_reason(
        output_bytes=initial_size,
        free_bytes=initial_free,
        max_bytes=args.max_bytes,
        min_free_bytes=args.min_free_bytes,
    )
    state: dict[str, object] = {
        "schema_version": "guarded-gdrive-download-v1",
        "status": "preflight",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "file_id": args.file_id,
        "output": str(output),
        "max_bytes": args.max_bytes,
        "min_free_bytes": args.min_free_bytes,
        "initial_output_bytes": initial_size,
        "initial_final_output_bytes": initial_output_size,
        "initial_temporary_bytes": initial_temporary_size,
        "initial_free_bytes": initial_free,
        "resume_enabled": True,
    }
    if initial_reason:
        state.update(status="rejected_preflight_guard", guard_reason=initial_reason)
        write_state(state_path, state)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(gdown_pythonpath)
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--fuzzy",
        f"https://drive.google.com/file/d/{args.file_id}/view?usp=sharing",
        "-O",
        str(output),
        "--continue",
    ]
    state.update(status="running", command=command, updated_at=utc_now())
    write_state(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        state["pid"] = process.pid
        write_state(state_path, state)
        while process.poll() is None:
            size = output.stat().st_size if output.exists() else 0
            temporary_size = temporary_download_bytes(output)
            free = shutil.disk_usage(root).free
            incremental_size = effective_download_bytes(
                output_bytes=size,
                initial_free_bytes=initial_free,
                current_free_bytes=free,
            )
            effective_size = max(size, temporary_size, initial_size + incremental_size)
            reason = guard_reason(
                output_bytes=effective_size,
                free_bytes=free,
                max_bytes=args.max_bytes,
                min_free_bytes=args.min_free_bytes,
            )
            state.update(
                output_bytes=size,
                temporary_download_bytes=temporary_size,
                effective_download_bytes=effective_size,
                observed_disk_consumption_bytes=max(0, initial_free - free),
                free_bytes=free,
                updated_at=utc_now(),
            )
            write_state(state_path, state)
            if reason:
                stop_process_group(process)
                state.update(
                    status="stopped_by_resource_guard",
                    guard_reason=reason,
                    return_code=process.returncode,
                    output_bytes=output.stat().st_size if output.exists() else 0,
                    free_bytes=shutil.disk_usage(root).free,
                    updated_at=utc_now(),
                )
                write_state(state_path, state)
                return 2
            time.sleep(args.poll_seconds)

    final_size = output.stat().st_size if output.exists() else 0
    final_free = shutil.disk_usage(root).free
    if process.returncode != 0 or not output.is_file() or final_size == 0:
        state.update(
            status="download_failed_preserved",
            return_code=process.returncode,
            output_bytes=final_size,
            free_bytes=final_free,
            updated_at=utc_now(),
        )
        write_state(state_path, state)
        return 1
    state.update(
        status="complete_pending_archive_audit",
        return_code=0,
        output_bytes=final_size,
        free_bytes=final_free,
        sha256=sha256_file(output),
        updated_at=utc_now(),
    )
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
