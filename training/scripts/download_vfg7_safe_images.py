#!/usr/bin/env python3
"""Download a checksum-pinned VFG-7 evaluation subset in small concurrent batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_plan(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"repo_id", "revision", "repo_path", "sha256", "size", "destination"}
        if not required.issubset(row):
            raise ValueError(f"{path}:{line_number}: missing keys {sorted(required - set(row))}")
        rows.append(row)
    if len({row["repo_path"] for row in rows}) != len(rows):
        raise ValueError("download plan contains duplicate repository paths")
    return rows


def verify_existing(row: dict) -> bool:
    destination = Path(row["destination"])
    return (
        destination.is_file()
        and destination.stat().st_size == int(row["size"])
        and sha256(destination) == str(row["sha256"])
    )


def download_one(row: dict, endpoint: str, retries: int) -> dict:
    destination = Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row["size"])
    expected_sha = str(row["sha256"])
    if verify_existing(row):
        return {"path": row["repo_path"], "size": expected_size, "status": "reused"}
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    encoded = urllib.parse.quote(str(row["repo_path"]), safe="/")
    url = f"{endpoint}/datasets/{row['repo_id']}/resolve/{row['revision']}/{encoded}?download=true"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url, stream=True, timeout=(30, 180), allow_redirects=True,
                headers={"User-Agent": "vcas-vfg7-safe-eval/1.0"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise IOError(f"size mismatch {actual_size} != {expected_size}")
            actual_sha = sha256(temporary)
            if actual_sha != expected_sha:
                raise IOError(f"sha256 mismatch {actual_sha} != {expected_sha}")
            os.replace(temporary, destination)
            return {"path": row["repo_path"], "size": expected_size, "status": "downloaded"}
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{row['repo_path']}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    rows = load_plan(args.plan)
    workers = max(1, min(args.workers, 16))
    batch_size = max(workers, min(args.batch_size, 256))
    args.state.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = sum(int(row["size"]) for row in rows)
    state = {
        "schema_version": "vfg7-safe-image-download-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256(args.plan),
        "endpoint": args.endpoint,
        "workers": workers,
        "batch_size": batch_size,
        "files_total": len(rows),
        "bytes_total": expected_bytes,
        "files_verified": 0,
        "bytes_verified": 0,
        "license": "CC BY-NC 4.0",
        "training_eligible": False,
        "purpose": "independent vehicle type/color validation and withheld testing only",
        "frozen_video_used": False,
    }
    atomic_json(args.state, state)
    completed: dict[str, dict] = {}
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(download_one, row, args.endpoint, args.retries): row
                    for row in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    completed[result["path"]] = result
            verified_bytes = sum(item["size"] for item in completed.values())
            state.update(
                updated_at=now(), files_verified=len(completed), bytes_verified=verified_bytes,
                progress=len(completed) / len(rows) if rows else 1.0,
            )
            atomic_json(args.state, state)
            print(json.dumps({
                "files_verified": len(completed), "files_total": len(rows),
                "bytes_verified": verified_bytes, "bytes_total": expected_bytes,
            }), flush=True)
        # Full end-to-end verification catches external modification and reused-file errors.
        for row in rows:
            if not verify_existing(row):
                raise RuntimeError(f"final verification failed: {row['repo_path']}")
        state.update(
            status="complete_checksum_verified", updated_at=now(), files_verified=len(rows),
            bytes_verified=expected_bytes, progress=1.0,
            downloaded_files=sum(item["status"] == "downloaded" for item in completed.values()),
            reused_files=sum(item["status"] == "reused" for item in completed.values()),
        )
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
