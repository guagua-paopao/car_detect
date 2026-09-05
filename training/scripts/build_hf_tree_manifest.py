#!/usr/bin/env python3
"""Build a resumable file manifest from a Hugging Face repository tree."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


_original_getaddrinfo = socket.getaddrinfo


def ipv4_getaddrinfo(
    host: str,
    port: int | str,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[tuple]:
    """Force IPv4 because this cloud container has no usable IPv6 address."""
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = ipv4_getaddrinfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--repo", default="iisc-aim/BMD-45")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def next_link(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">", start + 1)
        if start >= 0 and end > start:
            return part[start + 1 : end]
    return None


def fetch_page(url: str) -> tuple[list[dict], str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "VCAS-BMD45/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
                return payload, next_link(response.headers.get("Link"))
        except Exception as exc:  # noqa: BLE001 - retries report the final error
            last_error = exc
            print(
                f"manifest request attempt={attempt}/8 failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(attempt * 5)
    raise RuntimeError(f"failed to fetch repository tree page: {url}") from last_error


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")
    repo = urllib.parse.quote(args.repo, safe="/")
    revision = urllib.parse.quote(args.revision, safe="")
    url: str | None = (
        f"{endpoint}/api/datasets/{repo}/tree/{revision}"
        "?recursive=true&expand=false&limit=1000"
    )
    files: dict[str, tuple[int, str, str]] = {}

    while url:
        print(f"manifest request: {url}", flush=True)
        page, next_url = fetch_page(url)
        if next_url:
            parsed_next = urllib.parse.urlsplit(next_url)
            url = endpoint + parsed_next.path
            if parsed_next.query:
                url += "?" + parsed_next.query
        else:
            url = None
        for item in page:
            if item.get("type") != "file":
                continue
            path = str(item["path"])
            size = int(item.get("size") or 0)
            lfs = item.get("lfs") or {}
            sha256 = str(lfs.get("oid") or "-")
            encoded_path = urllib.parse.quote(path, safe="/")
            files[path] = (size, sha256, encoded_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(files):
            size, sha256, encoded_path = files[path]
            handle.write(f"{path}\t{size}\t{sha256}\t{encoded_path}\n")
    print(f"manifest={args.output} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
