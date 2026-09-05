"""Resumable, bounded-concurrency UVH-26 subset downloader."""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fetch(item: tuple[str, str, int], mirror: str, root: Path) -> tuple[str, str, int]:
    remote, relative, expected = item
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected or destination.stat().st_size == expected):
        return remote, "skipped", destination.stat().st_size
    part = destination.with_suffix(destination.suffix + ".part")
    url = mirror.rstrip("/") + "/datasets/iisc-aim/UVH-26/resolve/main/" + urllib.parse.quote(remote, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "vcas-uvh26-subset/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if expected and part.stat().st_size != expected:
        raise RuntimeError(f"size mismatch {remote}: {part.stat().st_size} != {expected}")
    os.replace(part, destination)
    return remote, "downloaded", destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    items = []
    with args.list.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"invalid download row: {row}")
            items.append((row[0], row[1], int(row[2])))
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch, item, args.mirror, args.output_root): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                remote, status, size = future.result()
                results.append({"remote": remote, "status": status, "bytes": size})
            except Exception as exc:  # fail closed: report every failed file
                failures.append({"remote": item[0], "error": repr(exc)})
    report = {"schema_version": "uvh26-download-v1", "source": "iisc-aim/UVH-26", "requested": len(items), "completed": len(results), "failed": len(failures), "results": sorted(results, key=lambda x: x["remote"]), "failures": sorted(failures, key=lambda x: x["remote"])}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("requested", "completed", "failed")}, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
