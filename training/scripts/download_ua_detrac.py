#!/usr/bin/env python3
"""Download and checksum the pinned UA-DETRAC Hugging Face mirror.

The downloader intentionally uses two file-level workers at most.  This avoids
the socket exhaustion seen with an unconstrained snapshot download while still
allowing the train and test archives to transfer concurrently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO_ID = "abhineet123/ua_detrac"
REVISION = "72045f434a646e6dc9b04e251a4108705cdfa5bf"
FILES = {
    "ua_detrac_training_set.zip": {
        "size": 5_630_247_258,
        "sha256": "7045ffc28bb247248e2c16f8329c7131751fe1135bc96e327aef8816508ffaf9",
    },
    "ua_detrac_test_set.zip": {
        "size": 4_252_838_102,
        "sha256": "ab018f74b2ed84bee47593871771887b771c7d79b6fb4fea0be5a89bfba77edd",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(path: Path, state: dict, lock: threading.Lock) -> None:
    with lock:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 2:
        raise ValueError("workers must be between 1 and 2")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "download-state.json"
    lock = threading.Lock()
    state = {
        "schema_version": "ua-detrac-download-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "endpoint": args.endpoint,
        "workers": args.workers,
        "frozen_video_used": False,
        "purpose": "independent trajectory/type/color stability evaluation; no production deployment authorization",
        "license": {
            "mirror_declared": "CC BY 4.0",
            "upstream_caveat": "The upstream project page does not expose an explicit license text; legal review is required before production training use.",
        },
        "files": {
            name: {**metadata, "status": "pending", "downloaded_path": None, "actual_sha256": None}
            for name, metadata in FILES.items()
        },
    }
    write_state(state_path, state, lock)

    def fetch(name: str, metadata: dict) -> tuple[str, Path, str]:
        target = args.output / name
        with lock:
            state["files"][name]["status"] = "downloading"
            state["files"][name]["started_at"] = now()
            state["updated_at"] = now()
        write_state(state_path, state, lock)

        if target.exists() and target.stat().st_size == metadata["size"]:
            digest = file_sha256(target)
            if digest == metadata["sha256"]:
                return name, target, digest
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REVISION,
                filename=name,
                local_dir=args.output,
                endpoint=args.endpoint,
                force_download=False,
            )
        )
        if downloaded.stat().st_size != metadata["size"]:
            raise RuntimeError(
                f"size mismatch for {name}: {downloaded.stat().st_size} != {metadata['size']}"
            )
        digest = file_sha256(downloaded)
        if digest != metadata["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {name}: {digest} != {metadata['sha256']}")
        return name, downloaded, digest

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, name, metadata): name for name, metadata in FILES.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, downloaded, digest = future.result()
                with lock:
                    state["files"][name].update(
                        {
                            "status": "complete",
                            "downloaded_path": str(downloaded),
                            "actual_sha256": digest,
                            "completed_at": now(),
                        }
                    )
                print(json.dumps({"file": name, "status": "complete", "sha256": digest}), flush=True)
            except Exception as error:
                failures.append(name)
                with lock:
                    state["files"][name].update(
                        {"status": "failed", "error": f"{type(error).__name__}: {error}", "failed_at": now()}
                    )
                print(json.dumps({"file": name, "status": "failed", "error": repr(error)}), flush=True)
            finally:
                with lock:
                    state["updated_at"] = now()
                write_state(state_path, state, lock)

    state["status"] = "failed" if failures else "complete"
    state["updated_at"] = now()
    state["failed_files"] = failures
    write_state(state_path, state, lock)
    print(json.dumps({"status": state["status"], "state": str(state_path)}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
