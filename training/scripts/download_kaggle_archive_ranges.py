#!/usr/bin/env python3
"""Download a public Kaggle archive with resumable checksum-verified ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_archive(ref: str, version: int) -> tuple[str, int, str]:
    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}"
    response = requests.get(api_url, allow_redirects=False, timeout=60, headers={"User-Agent": "vcas-kaggle-audit/1.0"})
    if response.status_code not in {301, 302, 303, 307, 308} or "Location" not in response.headers:
        raise RuntimeError(f"Kaggle archive did not redirect: {response.status_code}")
    url = response.headers["Location"]
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    if probe.status_code != 206:
        raise RuntimeError(f"archive does not support byte ranges: {probe.status_code}")
    match = re.fullmatch(r"bytes 0-0/([0-9]+)", probe.headers.get("Content-Range", ""))
    if not match:
        raise RuntimeError(f"unexpected Content-Range: {probe.headers.get('Content-Range')}")
    etag = probe.headers.get("ETag", "").strip('"').lower()
    if not re.fullmatch(r"[0-9a-f]{32}", etag):
        raise RuntimeError(f"expected a GCS MD5 ETag, got {etag!r}")
    return url, int(match.group(1)), etag


def fetch(url: str, start: int, end: int, path: Path, retries: int) -> dict:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return {"start": start, "end": end, "size": expected, "status": "reused"}
    if path.exists() and not path.is_file():
        raise RuntimeError(f"range target is not a regular file: {path}")
    last_error = None
    for attempt in range(1, retries + 1):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            existing = temporary.stat().st_size if temporary.is_file() else 0
            if existing > expected:
                raise IOError(f"partial range size {existing} exceeds expected {expected}")
            if existing == expected:
                os.replace(temporary, path)
                return {"start": start, "end": end, "size": expected, "status": "resumed"}
            request_start = start + existing
            with requests.get(
                url, headers={"Range": f"bytes={request_start}-{end}", "User-Agent": "vcas-kaggle-audit/1.0"},
                stream=True, timeout=(30, 180), allow_redirects=True,
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range status {response.status_code}")
                if not response.headers.get("Content-Range", "").startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range {response.headers.get('Content-Range')}")
                with temporary.open("ab" if existing else "wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            if temporary.stat().st_size != expected:
                raise IOError(f"range size {temporary.stat().st_size} != {expected}")
            os.replace(temporary, path)
            return {"start": start, "end": end, "size": expected, "status": "downloaded"}
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    url, expected_size, expected_md5 = resolve_archive(args.dataset_ref, args.version)
    if args.output.is_file():
        actual_size = args.output.stat().st_size
        actual_md5 = digest(args.output, "md5") if actual_size == expected_size else ""
        if actual_size == expected_size and actual_md5 == expected_md5:
            state = {
                "schema_version": "kaggle-range-download-v1", "status": "complete_checksum_verified",
                "updated_at": now(), "dataset_ref": args.dataset_ref, "version": args.version,
                "output": str(args.output.resolve()), "bytes_total": expected_size,
                "md5": actual_md5, "sha256": digest(args.output, "sha256"),
                "license": args.license, "frozen_video_used": False,
            }
            atomic_json(args.state, state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        raise FileExistsError(f"existing output does not match expected size/MD5: {args.output}")
    if args.output.exists():
        raise RuntimeError(f"output exists but is not a regular file: {args.output}")

    chunk_bytes = max(1, args.chunk_mib) * 1024 * 1024
    parts_root = args.output.with_name(args.output.name + ".range-parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = []
    start = 0
    while start < expected_size:
        end = min(expected_size - 1, start + chunk_bytes - 1)
        ranges.append((start, end, parts_root / f"{start:012d}-{end:012d}.part"))
        start = end + 1
    state = {
        "schema_version": "kaggle-range-download-v1", "status": "downloading",
        "created_at": now(), "updated_at": now(), "dataset_ref": args.dataset_ref,
        "version": args.version, "output": str(args.output.resolve()),
        "bytes_total": expected_size, "expected_md5_from_gcs_etag": expected_md5,
        "chunk_bytes": chunk_bytes, "chunks_total": len(ranges), "chunks_complete": 0,
        "workers": max(1, min(args.workers, 16)), "license": args.license,
        "purpose": "vehicle color/type training-source audit", "frozen_video_used": False,
    }
    atomic_json(args.state, state)
    try:
        completed = []
        with ThreadPoolExecutor(max_workers=state["workers"]) as executor:
            futures = {executor.submit(fetch, url, begin, end, path, args.retries): path for begin, end, path in ranges}
            for result in as_completed(futures):
                completed.append(result.result())
                state.update(
                    updated_at=now(), chunks_complete=len(completed),
                    bytes_downloaded=sum(item["size"] for item in completed),
                    progress=len(completed) / len(ranges),
                )
                atomic_json(args.state, state)
                if len(completed) % 4 == 0 or len(completed) == len(ranges):
                    print(json.dumps({
                        "chunks_complete": len(completed), "chunks_total": len(ranges),
                        "bytes_downloaded": state["bytes_downloaded"], "bytes_total": expected_size,
                    }), flush=True)
        assembling = args.output.with_name(args.output.name + ".assembling")
        if assembling.exists():
            raise FileExistsError(f"refusing to overwrite stale assembly file: {assembling}")
        state.update(status="assembling", updated_at=now())
        atomic_json(args.state, state)
        with assembling.open("wb") as destination:
            for begin, end, path in ranges:
                if not path.is_file() or path.stat().st_size != end - begin + 1:
                    raise RuntimeError(f"missing or invalid range part: {path}")
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        if assembling.stat().st_size != expected_size:
            raise RuntimeError("assembled size mismatch")
        state.update(status="hashing", updated_at=now())
        atomic_json(args.state, state)
        actual_md5 = digest(assembling, "md5")
        actual_sha256 = digest(assembling, "sha256")
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch {actual_md5} != {expected_md5}")
        os.replace(assembling, args.output)
        state.update(
            status="complete_checksum_verified", updated_at=now(), progress=1.0,
            bytes_downloaded=expected_size, md5=actual_md5, sha256=actual_sha256,
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
