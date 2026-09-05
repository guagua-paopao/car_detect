#!/usr/bin/env python3
"""Download one immutable HTTP object with resumable concurrent byte ranges."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from download_kaggle_archive_ranges import atomic_json, digest, fetch, now


def resolve_object(url: str) -> tuple[int, str]:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "vcas-http-range-audit/1.0"},
        timeout=(30, 120),
        allow_redirects=True,
    )
    if response.status_code != 206:
        raise RuntimeError(f"object does not support byte ranges: {response.status_code}")
    match = re.fullmatch(r"bytes 0-0/([0-9]+)", response.headers.get("Content-Range", ""))
    if not match:
        raise RuntimeError(f"unexpected Content-Range: {response.headers.get('Content-Range')}")
    etag = response.headers.get("ETag", "").strip('"').lower()
    if not re.fullmatch(r"[0-9a-f]{32}", etag):
        raise RuntimeError(f"expected immutable MD5 ETag, got {etag!r}")
    return int(match.group(1)), etag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--license", required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_md5 = resolve_object(args.url)
    if args.output.is_file():
        actual_size = args.output.stat().st_size
        actual_md5 = digest(args.output, "md5") if actual_size == expected_size else ""
        if actual_size == expected_size and actual_md5 == expected_md5:
            state = {
                "schema_version": "http-range-download-v1",
                "status": "complete_checksum_verified",
                "updated_at": now(),
                "url": args.url,
                "output": str(args.output.resolve()),
                "bytes_total": expected_size,
                "md5": actual_md5,
                "sha256": digest(args.output, "sha256"),
                "license": args.license,
                "purpose": args.purpose,
                "frozen_video_used": False,
            }
            atomic_json(args.state, state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        raise FileExistsError("existing output does not match expected size/MD5")
    if args.output.exists():
        raise RuntimeError("output exists but is not a regular file")

    chunk_bytes = max(1, args.chunk_mib) * 1024 * 1024
    parts_root = args.output.with_name(args.output.name + ".range-parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = []
    for start in range(0, expected_size, chunk_bytes):
        end = min(expected_size - 1, start + chunk_bytes - 1)
        ranges.append((start, end, parts_root / f"{start:012d}-{end:012d}.part"))
    workers = max(1, min(args.workers, 12))
    state = {
        "schema_version": "http-range-download-v1",
        "status": "downloading",
        "created_at": now(),
        "updated_at": now(),
        "url": args.url,
        "output": str(args.output.resolve()),
        "bytes_total": expected_size,
        "expected_md5_from_etag": expected_md5,
        "chunk_bytes": chunk_bytes,
        "chunks_total": len(ranges),
        "chunks_complete": 0,
        "workers": workers,
        "license": args.license,
        "purpose": args.purpose,
        "frozen_video_used": False,
    }
    atomic_json(args.state, state)
    completed = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch, args.url, start, end, path, args.retries): path
                for start, end, path in ranges
            }
            for future in as_completed(futures):
                completed.append(future.result())
                state.update(
                    updated_at=now(),
                    chunks_complete=len(completed),
                    bytes_downloaded=sum(item["size"] for item in completed),
                    progress=len(completed) / len(ranges),
                )
                atomic_json(args.state, state)
                if len(completed) % 8 == 0 or len(completed) == len(ranges):
                    print(json.dumps({
                        "chunks_complete": len(completed),
                        "chunks_total": len(ranges),
                        "bytes_downloaded": state["bytes_downloaded"],
                        "bytes_total": expected_size,
                    }), flush=True)
        assembling = args.output.with_name(args.output.name + ".assembling")
        if assembling.exists():
            raise FileExistsError(f"refusing to overwrite stale assembly file: {assembling}")
        state.update(status="assembling", updated_at=now())
        atomic_json(args.state, state)
        with assembling.open("wb") as destination:
            for start, end, part in ranges:
                if not part.is_file() or part.stat().st_size != end - start + 1:
                    raise RuntimeError(f"missing or invalid part: {part}")
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        state.update(status="hashing", updated_at=now())
        atomic_json(args.state, state)
        actual_md5 = digest(assembling, "md5")
        actual_sha256 = digest(assembling, "sha256")
        if assembling.stat().st_size != expected_size or actual_md5 != expected_md5:
            raise RuntimeError("assembled object failed size/MD5 verification")
        os.replace(assembling, args.output)
        state.update(
            status="complete_checksum_verified",
            updated_at=now(),
            progress=1.0,
            bytes_downloaded=expected_size,
            md5=actual_md5,
            sha256=actual_sha256,
            range_parts_preserved=True,
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
