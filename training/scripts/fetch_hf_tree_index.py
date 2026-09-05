"""Fetch a paginated Hugging Face dataset tree through a mirror.

The mirror is configurable because the training host may not reach the
canonical Hugging Face endpoint.  Only metadata is fetched; image content is
downloaded separately by the UVH subset sampler.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


def next_link(headers) -> str | None:
    link = headers.get("Link", "")
    for part in link.split(","):
        if 'rel="next"' in part:
            start, end = part.find("<"), part.find(">")
            if start >= 0 and end > start:
                return part[start + 1 : end]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args()
    url = f"{args.mirror.rstrip('/')}/api/datasets/{args.repo}/tree/main?recursive=true"
    records = []
    seen = set()
    pages = 0
    while url and pages < args.max_pages:
        if url.startswith("https://huggingface.co/"):
            url = args.mirror.rstrip("/") + url[len("https://huggingface.co") :]
        request = urllib.request.Request(url, headers={"User-Agent": "vcas-uvh26-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.loads(response.read())
            link = next_link(response.headers)
        if not isinstance(batch, list):
            raise RuntimeError(f"unexpected tree response at {url}")
        for item in batch:
            path = item.get("path")
            if path and path not in seen:
                seen.add(path)
                records.append(item)
        pages += 1
        url = link
    if url:
        raise RuntimeError(f"pagination exceeded --max-pages={args.max_pages}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": pages, "files": len(records), "png": sum(str(x.get("path", "")).endswith(".png") for x in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
