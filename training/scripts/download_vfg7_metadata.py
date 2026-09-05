#!/usr/bin/env python3
"""Download and checksum VFG-7 metadata/labels without downloading images."""

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


ROOT_FILES = ("LICENSE", "README.md", "README_en.md", "data.yaml", "vlm_annotations.json")


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


def list_tree(endpoint: str, repo_id: str, revision: str, prefix: str) -> list[dict]:
    encoded = urllib.parse.quote(prefix, safe="/")
    url = f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/{encoded}?recursive=true&expand=false&limit=1000"
    items = []
    while url:
        response = None
        last_error = None
        for attempt in range(1, 7):
            try:
                response = requests.get(url, timeout=60, headers={"User-Agent": "vcas-vfg7-audit/1.0"})
                response.raise_for_status()
                break
            except Exception as error:
                last_error = error
                time.sleep(min(2 ** attempt, 30))
        if response is None:
            raise RuntimeError(f"tree page failed after retries: {last_error}")
        page = response.json()
        if not isinstance(page, list):
            raise TypeError(f"unexpected tree response for {prefix}")
        items.extend(page)
        url = response.links.get("next", {}).get("url")
        if url:
            source = urllib.parse.urlsplit(url)
            mirror = urllib.parse.urlsplit(endpoint)
            url = urllib.parse.urlunsplit((mirror.scheme, mirror.netloc, source.path, source.query, source.fragment))
    return items


def download_one(endpoint: str, repo_id: str, revision: str, item: dict, root: Path) -> dict:
    relative = Path(str(item["path"]))
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    if destination.exists() and destination.stat().st_size == expected_size:
        return {"path": relative.as_posix(), "size": expected_size, "sha256": sha256(destination), "status": "reused"}
    encoded = urllib.parse.quote(relative.as_posix(), safe="/")
    url = f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{encoded}?download=true"
    last_error = None
    for attempt in range(1, 7):
        try:
            with requests.get(url, stream=True, timeout=(30, 120), headers={"User-Agent": "vcas-vfg7-audit/1.0"}) as response:
                response.raise_for_status()
                temporary = destination.with_suffix(destination.suffix + ".part")
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                if temporary.stat().st_size != expected_size:
                    raise IOError(f"size mismatch {temporary.stat().st_size} != {expected_size}")
                os.replace(temporary, destination)
                return {
                    "path": relative.as_posix(), "size": expected_size, "sha256": sha256(destination),
                    "source_oid": item.get("oid"), "status": "complete",
                }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed {relative}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--repo-id", default="Telody1220/VFG-7")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "metadata-download-state.json"
    state = {
        "schema_version": "vfg7-metadata-download-v1", "status": "running", "created_at": now(),
        "updated_at": now(), "repo_id": args.repo_id, "endpoint": args.endpoint,
        "frozen_video_used": False, "images_downloaded": False,
        "license": "CC BY-NC 4.0", "training_eligible": False,
        "purpose": "metadata alignment audit and independent evaluation planning only",
    }
    atomic_json(state_path, state)
    try:
        info = requests.get(
            f"{args.endpoint}/api/datasets/{args.repo_id}", timeout=60,
            headers={"User-Agent": "vcas-vfg7-audit/1.0"},
        )
        info.raise_for_status()
        revision = str(info.json()["sha"])
        state["resolved_revision"] = revision
        items = []
        for prefix in ("labels/train", "labels/val"):
            items.extend(item for item in list_tree(args.endpoint, args.repo_id, revision, prefix) if item.get("type") == "file")
        root_tree = list_tree(args.endpoint, args.repo_id, revision, "")
        root_by_path = {str(item.get("path")): item for item in root_tree if item.get("type") == "file"}
        for name in ROOT_FILES:
            if name not in root_by_path:
                raise RuntimeError(f"missing repository metadata file: {name}")
            items.append(root_by_path[name])
        unique = {str(item["path"]): item for item in items}
        results = []
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
            futures = {
                executor.submit(download_one, args.endpoint, args.repo_id, revision, item, args.output): path
                for path, item in unique.items()
            }
            for count, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if count % 250 == 0 or count == len(futures):
                    print(json.dumps({"verified": count, "total": len(futures)}), flush=True)
        results.sort(key=lambda item: item["path"])
        index_path = args.output / "source-index.json"
        atomic_json(index_path, {
            "schema_version": "vfg7-metadata-source-index-v1", "created_at": now(),
            "repo_id": args.repo_id, "resolved_revision": revision, "files": results,
        })
        state.update(
            status="complete", updated_at=now(), files=len(results),
            bytes=sum(item["size"] for item in results),
            train_label_files=sum(item["path"].startswith("labels/train/") for item in results),
            val_label_files=sum(item["path"].startswith("labels/val/") for item in results),
            source_index=str(index_path), source_index_sha256=sha256(index_path),
        )
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
