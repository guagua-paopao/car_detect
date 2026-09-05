#!/usr/bin/env python3
"""Download unique Open Images train images from an audited CSV plan.

Downloads run in small concurrent batches, resume partial files with HTTP Range,
and verify both byte size and the Open Images metadata MD5 before publication.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required for source integrity, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_plan(path: Path, output_root: Path, url_template: str) -> list[dict]:
    unique: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            image_id = row.get("image_id", "").strip()
            if not image_id:
                raise ValueError(f"{path}:{line_number}: empty image_id")
            license_url = row.get("license_url", "")
            if "creativecommons.org/licenses/by/2.0" not in license_url:
                raise ValueError(f"{path}:{line_number}: unexpected image license")
            try:
                expected_size = int(row.get("original_size_bytes") or 0)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid original size") from error
            expected_md5 = row.get("original_md5_base64", "").strip()
            if expected_size <= 0 or not expected_md5:
                raise ValueError(f"{path}:{line_number}: missing size or MD5")
            item = {
                "image_id": image_id,
                "url": url_template.format(image_id=image_id),
                "fallback_url": row.get("original_url", "").strip(),
                "destination": str((output_root / "images" / f"{image_id}.jpg").resolve()),
                "expected_size": expected_size,
                "expected_md5_base64": expected_md5,
                "license_url": license_url,
                "author": row.get("author", ""),
                "landing_url": row.get("landing_url", ""),
            }
            previous = unique.get(image_id)
            if previous and any(
                previous[key] != item[key]
                for key in ("expected_size", "expected_md5_base64", "license_url")
            ):
                raise ValueError(f"{path}:{line_number}: inconsistent metadata for {image_id}")
            unique[image_id] = item
    return [unique[key] for key in sorted(unique)]


def verify(path: Path, expected_size: int, expected_md5: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and md5_base64(path) == expected_md5
    )


def stream_attempt(url: str, temporary: Path, expected_size: int) -> None:
    current = temporary.stat().st_size if temporary.exists() else 0
    if current > expected_size:
        temporary.unlink()
        current = 0
    headers = {"User-Agent": "vcas-openimages-plan/1.0"}
    if current:
        headers["Range"] = f"bytes={current}-"
    with requests.get(
        url,
        stream=True,
        timeout=(30, 180),
        allow_redirects=True,
        headers=headers,
    ) as response:
        response.raise_for_status()
        if current and response.status_code != 206:
            temporary.unlink(missing_ok=True)
            current = 0
        mode = "ab" if current and response.status_code == 206 else "wb"
        with temporary.open(mode) as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)


def download_one(item: dict, retries: int) -> dict:
    destination = Path(item["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["expected_size"])
    expected_md5 = item["expected_md5_base64"]
    if verify(destination, expected_size, expected_md5):
        return {"image_id": item["image_id"], "size": expected_size, "status": "reused"}
    temporary = destination.with_suffix(destination.suffix + ".part")
    if verify(temporary, expected_size, expected_md5):
        os.replace(temporary, destination)
        return {"image_id": item["image_id"], "size": expected_size, "status": "resumed"}
    urls = [item["url"]]
    if item["fallback_url"] and item["fallback_url"] not in urls:
        urls.append(item["fallback_url"])
    errors = []
    for attempt in range(1, retries + 1):
        url = urls[(attempt - 1) % len(urls)]
        try:
            stream_attempt(url, temporary, expected_size)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise IOError(f"size mismatch {actual_size} != {expected_size}")
            actual_md5 = md5_base64(temporary)
            if actual_md5 != expected_md5:
                temporary.unlink(missing_ok=True)
                raise IOError(f"MD5 mismatch {actual_md5} != {expected_md5}")
            os.replace(temporary, destination)
            return {
                "image_id": item["image_id"],
                "size": expected_size,
                "status": "downloaded",
            }
        except Exception as error:  # keep partial data for safe resume
            errors.append(f"{type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{item['image_id']}: {' | '.join(errors[-3:])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--url-template",
        default="https://storage.googleapis.com/openimages/2018_04/train/{image_id}.jpg",
    )
    args = parser.parse_args()
    workers = max(1, min(args.workers, 12))
    batch_size = max(workers, min(args.batch_size, 128))
    items = load_plan(args.plan, args.output_root, args.url_template)
    if not items:
        raise RuntimeError("download plan contains no images")
    expected_bytes = sum(item["expected_size"] for item in items)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "openimages-plan-download-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256(args.plan),
        "output_root": str(args.output_root.resolve()),
        "workers": workers,
        "batch_size": batch_size,
        "files_total": len(items),
        "bytes_total": expected_bytes,
        "files_verified": 0,
        "bytes_verified": 0,
        "license": "per-image CC BY 2.0 verified in audited source plan",
        "training_eligible": True,
        "split": "train",
        "frozen_video_used": False,
    }
    atomic_json(args.state, state)
    completed: dict[str, dict] = {}
    failures: dict[str, str] = {}
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(download_one, item, args.retries): item for item in batch
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        result = future.result()
                        completed[result["image_id"]] = result
                    except Exception as error:
                        failures[item["image_id"]] = f"{type(error).__name__}: {error}"
            verified_bytes = sum(result["size"] for result in completed.values())
            state.update(
                updated_at=now(),
                files_verified=len(completed),
                bytes_verified=verified_bytes,
                progress=len(completed) / len(items),
                failures=len(failures),
            )
            atomic_json(args.state, state)
            print(json.dumps({
                "files_verified": len(completed),
                "files_total": len(items),
                "bytes_verified": verified_bytes,
                "bytes_total": expected_bytes,
                "failures": len(failures),
            }), flush=True)
        # Re-check all published files; a partial batch is not a valid completed dataset.
        invalid = [
            item["image_id"]
            for item in items
            if not verify(
                Path(item["destination"]),
                int(item["expected_size"]),
                item["expected_md5_base64"],
            )
        ]
        if failures or invalid:
            state.update(
                status="incomplete_fail_closed",
                updated_at=now(),
                failures=len(failures),
                failure_examples=dict(list(sorted(failures.items()))[:20]),
                invalid_files=invalid[:20],
            )
            atomic_json(args.state, state)
            raise RuntimeError(
                f"download incomplete: failures={len(failures)}, invalid={len(invalid)}"
            )
        state.update(
            status="complete_checksum_verified",
            updated_at=now(),
            files_verified=len(items),
            bytes_verified=expected_bytes,
            progress=1.0,
            downloaded_files=sum(r["status"] == "downloaded" for r in completed.values()),
            reused_files=sum(r["status"] == "reused" for r in completed.values()),
            resumed_files=sum(r["status"] == "resumed" for r in completed.values()),
            failures=0,
        )
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        if state.get("status") == "running":
            state.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
            atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
