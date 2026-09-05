#!/usr/bin/env python3
"""Run the UA-DETRAC track-manifest builder after verified extraction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--annotation-source", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()
    state_path = args.source / "track-build-state.json"
    state = {
        "schema_version": "ua-detrac-track-build-runner-v1",
        "status": "waiting_for_extraction",
        "created_at": now(),
        "updated_at": now(),
        "source": str(args.source),
        "output": str(args.output),
        "frozen_video_used": False,
    }
    atomic_json(state_path, state)
    deadline = time.monotonic() + args.timeout_hours * 3600
    extraction_state_path = args.source / "extraction-state.json"
    annotation_state_path = args.annotation_source / "annotation-download-state.json"
    while True:
        if time.monotonic() > deadline:
            state.update(status="failed", error="extraction wait timeout", updated_at=now())
            atomic_json(state_path, state)
            return 2
        if extraction_state_path.exists():
            extraction = json.loads(extraction_state_path.read_text(encoding="utf-8"))
            if extraction.get("status") == "failed":
                state.update(status="failed", error="extraction failed", updated_at=now())
                atomic_json(state_path, state)
                return 2
            if extraction.get("status") == "complete" and annotation_state_path.exists():
                annotation = json.loads(annotation_state_path.read_text(encoding="utf-8"))
                if annotation.get("status") == "failed":
                    state.update(status="failed", error="annotation download failed", updated_at=now())
                    atomic_json(state_path, state)
                    return 2
                if annotation.get("status") == "complete":
                    break
        time.sleep(max(5, args.poll_seconds))
    if args.output.exists() and any(args.output.iterdir()):
        state.update(status="failed", error=f"output is non-empty: {args.output}", updated_at=now())
        atomic_json(state_path, state)
        return 2
    state.update(status="building", updated_at=now())
    atomic_json(state_path, state)
    command = [
        sys.executable,
        str(args.builder),
        "--training-root", str(args.source / "extracted" / "training"),
        "--test-root", str(args.source / "extracted" / "test"),
        "--training-xml-root", str(args.annotation_source / "training"),
        "--test-xml-root", str(args.annotation_source / "test"),
        "--output-root", str(args.output),
        "--window", "5",
        "--windows-per-track", "2",
        "--minimum-track-agreement", "0.8",
        "--margin", "0.08",
        "--workers", "8",
    ]
    state["command"] = command
    atomic_json(state_path, state)
    completed = subprocess.run(command, check=False)
    state.update(
        status="complete" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        updated_at=now(),
    )
    atomic_json(state_path, state)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
