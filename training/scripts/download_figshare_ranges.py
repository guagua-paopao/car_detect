#!/usr/bin/env python3
"""Download one pinned Figshare file with resumable ranges and official MD5 verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from download_kaggle_archive_ranges import atomic_json, digest, fetch, now


def validate_probe(content_range: str, expected_size: int, etag: str) -> dict[str, str | int]:
    match = re.fullmatch(r"bytes 0-0/([0-9]+)", content_range)
    if not match:
        raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
    observed_size = int(match.group(1))
    if observed_size != expected_size:
        raise RuntimeError(f"official size mismatch: {observed_size} != {expected_size}")
    normalized_etag = etag.strip().strip('"').lower()
    if not re.fullmatch(r"[0-9a-f]{32}(?:-[1-9][0-9]*)?", normalized_etag):
        raise RuntimeError(f"unexpected Figshare ETag: {etag!r}")
    return {"observed_size": observed_size, "etag": normalized_etag}


def probe(url: str, expected_size: int) -> dict[str, str | int]:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "vcas-figshare-range-audit/1.0"},
        timeout=(30, 120),
        allow_redirects=True,
    )
    if response.status_code != 206:
        raise RuntimeError(f"object does not support byte ranges: {response.status_code}")
    result = validate_probe(
        response.headers.get("Content-Range", ""), expected_size,
        response.headers.get("ETag", ""),
    )
    result["resolved_url"] = response.url
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--minimum-final-free-bytes", type=int, default=100 * 1024 ** 3)
    parser.add_argument("--maximum-object-bytes", type=int, default=16 * 1024 ** 3)
    parser.add_argument("--license", required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    expected_md5 = args.expected_md5.lower()
    if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        raise ValueError("expected MD5 must be 32 hexadecimal characters")
    if args.expected_size <= 0 or args.expected_size > args.maximum_object_bytes:
        raise ValueError("expected size violates object-size guard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output.parent).free
    required_free = args.minimum_final_free_bytes + 2 * args.expected_size
    if free_bytes < required_free:
        raise RuntimeError(f"free-space guard failed: {free_bytes} < {required_free}")
    probe_result = probe(args.url, args.expected_size)

    if args.output.is_file():
        actual_md5 = digest(args.output, "md5") if args.output.stat().st_size == args.expected_size else ""
        if args.output.stat().st_size == args.expected_size and actual_md5 == expected_md5:
            state = {
                "schema_version": "figshare-range-download-v1",
                "status": "complete_checksum_verified",
                "updated_at": now(),
                "url": args.url,
                "resolved_url": probe_result["resolved_url"],
                "output": str(args.output.resolve()),
                "bytes_total": args.expected_size,
                "official_md5": expected_md5,
                "md5": actual_md5,
                "sha256": digest(args.output, "sha256"),
                "license": args.license,
                "purpose": args.purpose,
                "frozen_video_used": False,
            }
            atomic_json(args.state, state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        raise FileExistsError("existing output does not match official size/MD5")
    if args.output.exists():
        raise RuntimeError("output exists but is not a regular file")

    chunk_bytes = max(1, args.chunk_mib) * 1024 * 1024
    workers = max(1, min(args.workers, 8))
    parts_root = args.output.with_name(args.output.name + ".range-parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(args.expected_size - 1, start + chunk_bytes - 1))
        for start in range(0, args.expected_size, chunk_bytes)
    ]
    state: dict[str, object] = {
        "schema_version": "figshare-range-download-v1",
        "status": "downloading",
        "created_at": now(),
        "updated_at": now(),
        "url": args.url,
        "resolved_url": probe_result["resolved_url"],
        "probe_etag": probe_result["etag"],
        "output": str(args.output.resolve()),
        "bytes_total": args.expected_size,
        "official_md5": expected_md5,
        "chunk_bytes": chunk_bytes,
        "chunks_total": len(ranges),
        "chunks_complete": 0,
        "workers": workers,
        "initial_free_bytes": free_bytes,
        "minimum_final_free_bytes": args.minimum_final_free_bytes,
        "maximum_object_bytes": args.maximum_object_bytes,
        "license": args.license,
        "purpose": args.purpose,
        "frozen_video_used": False,
    }
    atomic_json(args.state, state)
    completed: list[dict[str, object]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for start, end in ranges:
                path = parts_root / f"{start:012d}-{end:012d}.part"
                futures[executor.submit(fetch, args.url, start, end, path, args.retries)] = path
            for future in as_completed(futures):
                completed.append(future.result())
                state.update(
                    updated_at=now(),
                    chunks_complete=len(completed),
                    bytes_downloaded=sum(int(item["size"]) for item in completed),
                    progress=len(completed) / len(ranges),
                )
                atomic_json(args.state, state)
                if len(completed) % 8 == 0 or len(completed) == len(ranges):
                    print(json.dumps({
                        "chunks_complete": len(completed),
                        "chunks_total": len(ranges),
                        "bytes_downloaded": state["bytes_downloaded"],
                        "bytes_total": args.expected_size,
                    }), flush=True)

        assembling = args.output.with_name(args.output.name + ".assembling")
        if assembling.exists():
            raise FileExistsError(f"refusing to overwrite stale assembly file: {assembling}")
        state.update(status="assembling", updated_at=now())
        atomic_json(args.state, state)
        with assembling.open("wb") as destination:
            for start, end in ranges:
                part = parts_root / f"{start:012d}-{end:012d}.part"
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
        if assembling.stat().st_size != args.expected_size or actual_md5 != expected_md5:
            raise RuntimeError("assembled object failed official size/MD5 verification")
        os.replace(assembling, args.output)
        state.update(
            status="complete_checksum_verified",
            updated_at=now(),
            progress=1.0,
            bytes_downloaded=args.expected_size,
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
