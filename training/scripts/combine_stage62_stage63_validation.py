#!/usr/bin/env python3
"""Wait for both validation waves and build one fail-closed candidate ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_state(path: Path, session: str, timeout_hours: float) -> dict:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                return state
            if state.get("status") not in {"waiting_for_training", "running", "complete"}:
                raise RuntimeError(f"unexpected validation state for {path}: {state.get('status')}")
        alive = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not alive:
            raise RuntimeError(f"validation session {session} ended without complete evidence")
        time.sleep(30)
    raise TimeoutError(f"timed out waiting for {path}")


def checked_report(state: dict) -> tuple[Path, dict]:
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if state.get(key) is not False:
            raise RuntimeError(f"validation state policy violation: {key}")
    path = Path(state["report"])
    if not path.is_file() or sha256(path) != state.get("report_sha256"):
        raise RuntimeError("validation report is missing or hash-mismatched")
    report = json.loads(path.read_text(encoding="utf-8"))
    policy = report.get("policy", {})
    expected = {
        "validation_only": True,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    for key, value in expected.items():
        if policy.get(key) is not value:
            raise RuntimeError(f"validation report policy violation: {key}")
    return path, report


def rank_key(item: dict) -> tuple:
    gates = item.get("gates", {})
    hard = item.get("hard", {})
    comparison = item.get("comparison_to_production", {})
    return (
        item.get("screen_status") == "pass",
        sum(value is True for value in gates.values()),
        float(comparison.get("hard_body_coverage_gain_percentage_points", -999.0)),
        float(comparison.get("hard_color_unknown_relative_reduction", -999.0)),
        float(hard.get("body_type", {}).get("coverage", 0.0)),
        float(hard.get("color", {}).get("coverage", 0.0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage62-state", type=Path, required=True)
    parser.add_argument("--stage62-session", required=True)
    parser.add_argument("--stage63-state", type=Path, required=True)
    parser.add_argument("--stage63-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    args = parser.parse_args()
    if args.output.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite combined validation evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "attribute-stage62-63-combined-validation-run-v1",
        "status": "waiting",
        "created_at": now(),
        "updated_at": now(),
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage62_state = wait_state(args.stage62_state, args.stage62_session, args.timeout_hours)
    stage63_state = wait_state(args.stage63_state, args.stage63_session, args.timeout_hours)
    stage62_path, stage62 = checked_report(stage62_state)
    stage63_path, stage63 = checked_report(stage63_state)

    production = None
    candidates = []
    seen_names = set()
    for wave, report in (("stage62", stage62), ("stage63", stage63)):
        for raw in report.get("models", []):
            item = {**raw, "wave": wave}
            name = item["model"]
            if name == "production":
                if production is None:
                    production = item
                elif item.get("checkpoint_sha256") != production.get("checkpoint_sha256"):
                    raise RuntimeError("validation waves used different production baselines")
                continue
            if name in seen_names:
                raise RuntimeError(f"duplicate candidate name across validation waves: {name}")
            seen_names.add(name)
            candidates.append(item)
    if production is None or not candidates:
        raise RuntimeError("combined validation lacks production or candidates")

    ranked = sorted(candidates, key=rank_key, reverse=True)
    passing = [item for item in ranked if item.get("screen_status") == "pass"]
    failure_counts = Counter()
    for item in candidates:
        for gate, value in item.get("gates", {}).items():
            if value is not True:
                failure_counts[gate] += 1
    report = {
        "schema_version": "attribute-stage62-63-combined-validation-v1",
        "created_at": now(),
        "status": "pass_candidates_available" if passing else "complete_all_candidates_rejected",
        "source_reports": [
            {"wave": "stage62", "path": str(stage62_path), "sha256": sha256(stage62_path)},
            {"wave": "stage63", "path": str(stage63_path), "sha256": sha256(stage63_path)},
        ],
        "production": production,
        "ranked_candidates": ranked,
        "passing_candidates": [item["model"] for item in passing],
        "top_two_for_seed_robustness": [item["model"] for item in passing[:2]],
        "failure_gate_counts": dict(sorted(failure_counts.items())),
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
            "ranking_does_not_override_any_gate": True,
        },
        "decision": (
            "run two additional seeds for at most the top two fully passing candidates"
            if passing else
            "do not access test; use per-gate failure counts and stratified reports to design the next round"
        ),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    state.update({
        "status": "complete",
        "updated_at": now(),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "passing_candidates": report["passing_candidates"],
    })
    temporary_state = args.state.with_suffix(args.state.suffix + ".tmp")
    temporary_state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_state, args.state)
    print(json.dumps({"status": report["status"], "passing_candidates": report["passing_candidates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
