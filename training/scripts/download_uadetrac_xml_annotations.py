#!/usr/bin/env python3
"""Download and verify the small UA-DETRAC sequence XML annotation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


PREFIXES = {
    "training": "DETRAC/DETRAC-Train-Annotations-XML",
    "test": "DETRAC/DETRAC-Test-Annotations-XML",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request_json(url: str, timeout: int = 60) -> tuple[object, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "vcas-uadetrac-acquisition/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), dict(response.headers.items())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_xml(path: Path) -> str:
    root = ET.parse(path).getroot()
    sequence = root.attrib.get("name", "").strip()
    if not sequence.startswith("MVI_") or root.find("frame") is None:
        raise ValueError(f"not a UA-DETRAC sequence XML: {path}")
    return sequence


def download_one(endpoint: str, repo_id: str, revision: str, item: dict, destination: Path) -> dict:
    expected_size = int(item["size"])
    if destination.exists() and destination.stat().st_size == expected_size:
        sequence = validate_xml(destination)
        return {
            "path": str(destination), "sequence": sequence, "size": expected_size,
            "sha256": sha256(destination), "status": "reused_verified",
        }
    temporary = destination.with_suffix(destination.suffix + ".part")
    encoded_path = urllib.parse.quote(str(item["path"]), safe="/")
    url = f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{encoded_path}?download=true"
    last_error = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vcas-uadetrac-acquisition/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise IOError(f"size mismatch {actual_size} != {expected_size}")
            sequence = validate_xml(temporary)
            os.replace(temporary, destination)
            return {
                "path": str(destination), "sequence": sequence, "size": actual_size,
                "sha256": sha256(destination), "source_oid": item.get("oid"), "status": "complete",
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"failed after retries: {item['path']}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--repo-id", default="kalyan1729/trafficmanagementdataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "annotation-download-state.json"
    state = {
        "schema_version": "uadetrac-xml-download-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "endpoint": args.endpoint,
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "frozen_video_used": False,
        "purpose": "independent attribute and trajectory evaluation only",
        "license": {
            "mirror_declared": "CC BY 4.0",
            "upstream_caveat": "upstream license text not independently verified; training use disabled",
        },
        "partitions": {},
    }
    atomic_json(state_path, state)
    try:
        repo_info, _ = request_json(f"{args.endpoint}/api/datasets/{args.repo_id}")
        resolved_revision = str(repo_info.get("sha", args.revision)) if isinstance(repo_info, dict) else args.revision
        state["resolved_revision"] = resolved_revision
        jobs = []
        indexes = {}
        for partition, prefix in PREFIXES.items():
            query = urllib.parse.quote(prefix, safe="/")
            tree_url = (
                f"{args.endpoint}/api/datasets/{args.repo_id}/tree/{resolved_revision}/{query}"
                "?recursive=true&expand=false&limit=1000"
            )
            entries, headers = request_json(tree_url)
            if not isinstance(entries, list):
                raise TypeError(f"unexpected tree response for {partition}")
            xml_items = [item for item in entries if item.get("type") == "file" and str(item.get("path", "")).lower().endswith(".xml")]
            if not xml_items:
                raise RuntimeError(f"no XML files listed for {partition}")
            destination_root = args.output / partition
            destination_root.mkdir(parents=True, exist_ok=True)
            indexes[partition] = {
                "prefix": prefix,
                "tree_headers": {key: value for key, value in headers.items() if key.lower() in {"etag", "x-repo-commit"}},
                "listed_files": xml_items,
            }
            for item in xml_items:
                jobs.append((partition, item, destination_root / Path(str(item["path"])).name))
        results = {partition: [] for partition in PREFIXES}
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
            futures = {
                executor.submit(download_one, args.endpoint, args.repo_id, resolved_revision, item, destination): partition
                for partition, item, destination in jobs
            }
            for count, future in enumerate(as_completed(futures), 1):
                partition = futures[future]
                results[partition].append(future.result())
                if count % 10 == 0 or count == len(futures):
                    print(json.dumps({"verified": count, "total": len(futures)}), flush=True)
        for partition in PREFIXES:
            results[partition].sort(key=lambda item: item["sequence"])
            index_path = args.output / partition / "source-index.json"
            index_payload = {
                "schema_version": "uadetrac-xml-source-index-v1",
                "created_at": now(),
                "repo_id": args.repo_id,
                "resolved_revision": resolved_revision,
                "partition": partition,
                "source_tree": indexes[partition],
                "files": results[partition],
            }
            atomic_json(index_path, index_payload)
            state["partitions"][partition] = {
                "status": "complete",
                "files": len(results[partition]),
                "bytes": sum(item["size"] for item in results[partition]),
                "index": str(index_path),
                "index_sha256": sha256(index_path),
            }
        state.update(status="complete", updated_at=now())
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
