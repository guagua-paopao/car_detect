#!/usr/bin/env python3
"""Resume-safe HTTP range downloader using bounded parallel batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def ranges(total_size: int, part_size: int) -> list[tuple[int, int, int]]:
    if total_size <= 0 or part_size <= 0:
        raise ValueError("sizes must be positive")
    result = []
    index = 0
    for start in range(0, total_size, part_size):
        result.append((index, start, min(total_size - 1, start + part_size - 1)))
        index += 1
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_part(
    url: str,
    part_dir: Path,
    item: tuple[int, int, int],
    retries: int,
    timeout: int,
) -> dict[str, int | str]:
    index, start, end = item
    expected = end - start + 1
    target = part_dir / f"part-{index:05d}-{start}-{end}.bin"
    if target.is_file() and target.stat().st_size == expected:
        return {"index": index, "bytes": expected, "status": "reused"}
    temporary = target.with_suffix(target.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "VCAS-dataset-audit/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                written = 0
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if written != expected:
                raise RuntimeError(f"part size {written} != expected {expected}")
            os.replace(temporary, target)
            return {"index": index, "bytes": written, "status": "downloaded"}
        except Exception as error:  # network failures are retried and preserved in the log
            last_error = error
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"part {index} failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--part-size-mib", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise RuntimeError("workers must be between 1 and 16")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.parent / f".{output.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    plan = ranges(args.expected_size, args.part_size_mib * 1024 * 1024)

    started = datetime.now(timezone.utc)
    if not (output.is_file() and output.stat().st_size == args.expected_size):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_part,
                    args.url,
                    part_dir,
                    item,
                    args.retries,
                    args.timeout,
                ): item[0]
                for item in plan
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    json.dumps(
                        {"completed_parts": completed, "total_parts": len(plan), **result},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        assembling = output.with_suffix(output.suffix + ".assembling")
        written = 0
        with assembling.open("wb") as destination:
            for index, start, end in plan:
                part = part_dir / f"part-{index:05d}-{start}-{end}.bin"
                expected = end - start + 1
                if not part.is_file() or part.stat().st_size != expected:
                    raise RuntimeError(f"part missing or invalid during assembly: {part}")
                with part.open("rb") as source:
                    while True:
                        chunk = source.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)
                        written += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if written != args.expected_size:
            raise RuntimeError(f"assembled size {written} != expected {args.expected_size}")
        os.replace(assembling, output)

    report = {
        "schema_version": "http-range-download-report-v1",
        "status": "complete",
        "url": args.url,
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "part_size_mib": args.part_size_mib,
        "parts": len(plan),
        "workers": args.workers,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
