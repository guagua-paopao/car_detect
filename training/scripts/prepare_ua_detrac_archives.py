#!/usr/bin/env python3
"""Wait for verified UA-DETRAC downloads and safely extract both archives."""

from __future__ import annotations

import argparse
import json
import stat
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_members(archive: zipfile.ZipFile) -> tuple[int, int]:
    members = archive.infolist()
    total = 0
    for member in members:
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise RuntimeError(f"symbolic link is not allowed: {member.filename}")
        total += member.file_size
    return len(members), total


def extract_one(source: Path, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty extraction target: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        member_count, uncompressed_bytes = validate_members(archive)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"CRC failure in {source.name}: {bad_member}")
        archive.extractall(destination)
    return {
        "archive": str(source),
        "destination": str(destination),
        "members": member_count,
        "uncompressed_bytes": uncompressed_bytes,
        "status": "complete",
        "completed_at": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()

    download_state_path = args.source / "download-state.json"
    output_state_path = args.source / "extraction-state.json"
    state = {
        "schema_version": "ua-detrac-extraction-v1",
        "status": "waiting_for_verified_download",
        "created_at": now(),
        "updated_at": now(),
        "source": str(args.source),
        "archives_preserved": True,
        "frozen_video_used": False,
        "extractions": {},
    }
    atomic_json(output_state_path, state)
    deadline = time.monotonic() + args.timeout_hours * 3600

    while True:
        if time.monotonic() > deadline:
            state.update(status="failed", error="download wait timeout", updated_at=now())
            atomic_json(output_state_path, state)
            return 2
        if download_state_path.exists():
            download_state = json.loads(download_state_path.read_text(encoding="utf-8"))
            status_value = download_state.get("status")
            if status_value == "failed":
                state.update(status="failed", error="download state is failed", updated_at=now())
                atomic_json(output_state_path, state)
                return 2
            if status_value == "complete":
                for name, metadata in download_state.get("files", {}).items():
                    if metadata.get("status") != "complete":
                        raise RuntimeError(f"download marked complete but file is not complete: {name}")
                    if metadata.get("actual_sha256") != metadata.get("sha256"):
                        raise RuntimeError(f"download checksum evidence mismatch: {name}")
                break
        time.sleep(max(5, args.poll_seconds))

    state.update(status="extracting", updated_at=now())
    atomic_json(output_state_path, state)
    plans = {
        "training": args.source / "ua_detrac_training_set.zip",
        "test": args.source / "ua_detrac_test_set.zip",
    }
    try:
        for partition, archive in plans.items():
            if not archive.is_file():
                raise RuntimeError(f"verified archive is missing: {archive}")
            result = extract_one(archive, args.source / "extracted" / partition)
            state["extractions"][partition] = result
            state["updated_at"] = now()
            atomic_json(output_state_path, state)
            print(json.dumps({"partition": partition, **result}, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        atomic_json(output_state_path, state)
        raise

    state.update(status="complete", updated_at=now())
    atomic_json(output_state_path, state)
    print(json.dumps({"status": "complete", "state": str(output_state_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
