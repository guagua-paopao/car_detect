#!/usr/bin/env python3
"""Complete a sequential partial file with verified concurrent HTTP range chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_chunk(url: str, start: int, end: int, path: Path) -> dict:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return {"start": start, "end": end, "size": expected, "path": str(path), "status": "reused"}
    last_error = None
    for attempt in range(1, 11):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "vcas-range-downloader/1.0"},
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range response status {response.status_code}, expected 206")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                temporary = path.with_suffix(path.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                if temporary.stat().st_size != expected:
                    raise RuntimeError(f"chunk size {temporary.stat().st_size} != {expected}")
                os.replace(temporary, path)
                return {"start": start, "end": end, "size": expected, "path": str(path), "status": "complete"}
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"chunk {start}-{end} failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if not args.partial.is_file():
        raise FileNotFoundError(args.partial)
    initial_size = args.partial.stat().st_size
    if initial_size <= 0 or initial_size > args.expected_size:
        raise ValueError(f"invalid partial size: {initial_size}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    chunk_bytes = args.chunk_mib * 1024 * 1024
    parts_root = args.state.parent / (args.output.name + ".range-parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = []
    cursor = initial_size
    while cursor < args.expected_size:
        end = min(args.expected_size - 1, cursor + chunk_bytes - 1)
        ranges.append((cursor, end, parts_root / f"{cursor:012d}-{end:012d}.part"))
        cursor = end + 1
    state = {
        "schema_version": "http-range-completion-v1",
        "status": "downloading",
        "created_at": now(),
        "updated_at": now(),
        "url": args.url,
        "partial": str(args.partial),
        "output": str(args.output),
        "initial_size": initial_size,
        "expected_size": args.expected_size,
        "expected_sha256": args.expected_sha256,
        "chunk_bytes": chunk_bytes,
        "workers": args.workers,
        "chunks": len(ranges),
    }
    atomic_json(args.state, state)
    try:
        completed = []
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 12))) as executor:
            futures = {executor.submit(fetch_chunk, args.url, start, end, path): (start, end, path) for start, end, path in ranges}
            for count, future in enumerate(as_completed(futures), 1):
                completed.append(future.result())
                state.update(completed_chunks=count, updated_at=now())
                atomic_json(args.state, state)
                print(json.dumps({"completed_chunks": count, "total_chunks": len(ranges)}), flush=True)
        state.update(status="assembling", updated_at=now())
        atomic_json(args.state, state)
        with args.partial.open("ab") as destination:
            for start, end, part in sorted(ranges):
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        actual_size = args.partial.stat().st_size
        if actual_size != args.expected_size:
            raise RuntimeError(f"assembled size mismatch {actual_size} != {args.expected_size}")
        state.update(status="hashing", actual_size=actual_size, updated_at=now())
        atomic_json(args.state, state)
        actual_sha256 = sha256(args.partial)
        if actual_sha256 != args.expected_sha256:
            raise RuntimeError(f"sha256 mismatch {actual_sha256} != {args.expected_sha256}")
        os.replace(args.partial, args.output)
        state.update(status="complete", actual_sha256=actual_sha256, updated_at=now())
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
