#!/usr/bin/env python3
"""Fail-closed Stage106 component gate for an integrated attribute candidate.

This decision consumes validation-only Stage104 (color) and Stage105 (body)
states.  It never accesses test data or the frozen demo video.  Passing this
gate only authorizes the next independent-test/backend stage; it does not
authorize deployment or frozen-video replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLOR_GATES = (
    "color_static_precision",
    "color_static_coverage",
    "color_complex_unknown_reduction",
    "color_track_precision",
    "color_track_coverage",
    "color_track_stability",
)
BODY_GATES = (
    "body_static_precision",
    "body_static_coverage",
    "body_complex_coverage_gain",
    "body_track_precision",
    "body_track_coverage",
    "body_track_stability",
)
SAFE_FALSE_FIELDS = (
    "test_accessed",
    "frozen_video_used",
    "production_model_modified",
    "backend_gates_run",
    "deployment_performed",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path, component: str) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "complete_validation_only":
        raise ValueError(f"{component} state is not complete_validation_only")
    for field in SAFE_FALSE_FIELDS:
        if state.get(field) is not False:
            raise ValueError(f"unsafe {component} state: {field} must be false")
    return state


def select_color(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("decision") != "color_component_qualified":
        return None
    qualified = set(state.get("qualified_variants", []))
    candidates: list[dict[str, Any]] = []
    for name, item in state.get("variants", {}).items():
        gates = item.get("color_gates", {})
        all_pass = all(gates.get(gate) is True for gate in COLOR_GATES)
        if name in qualified and item.get("all_color_gates_pass") is True and all_pass:
            candidates.append({"variant": name, **item})
    if not candidates:
        raise ValueError("color state claims qualification without a fully passing variant")

    def rank(item: dict[str, Any]) -> tuple[float, ...]:
        static = item["static"]
        track = item["track_final"]
        return (
            float(item["complex_unknown_relative_reduction"]),
            float(static["coverage"]),
            float(track["coverage"]),
            float(static["precision"]),
            float(item["stability"]),
        )

    return max(candidates, key=rank)


def select_body(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("decision") != "body_subtype_threshold_qualified":
        return None
    qualified = set(state.get("qualified_variants", []))
    candidates: list[dict[str, Any]] = []
    for item in state.get("variants", []):
        gates = item.get("body_gates", {})
        all_pass = all(gates.get(gate) is True for gate in BODY_GATES)
        if (
            item.get("variant") in qualified
            and item.get("all_body_gates_pass") is True
            and all_pass
        ):
            candidates.append(item)
    if not candidates:
        raise ValueError("body state claims qualification without a fully passing variant")

    def rank(item: dict[str, Any]) -> tuple[float, ...]:
        static = item["static"]
        track = item["track_final"]
        return (
            float(item["complex_coverage_gain"]),
            float(static["coverage"]),
            float(track["coverage"]),
            float(static["precision"]),
            float(item["stability"]),
        )

    return max(candidates, key=rank)


def decide(color_state_path: Path, body_state_path: Path) -> dict[str, Any]:
    color_state = load_state(color_state_path, "color")
    body_state = load_state(body_state_path, "body")
    color = select_color(color_state)
    body = select_body(body_state)

    failures: list[str] = []
    if color is None:
        failures.append("stage104_color_component_not_qualified")
    if body is None:
        failures.append("stage105_body_component_not_qualified")

    return {
        "schema_version": "stage106-integrated-component-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_for_integrated_independent_test"
            if not failures
            else "component_repair_required_fail_closed"
        ),
        "selected_color": color,
        "selected_body": body,
        "failure_reasons": failures,
        "evidence": {
            "stage104_state": str(color_state_path),
            "stage104_state_sha256": sha256(color_state_path),
            "stage105_state": str(body_state_path),
            "stage105_state_sha256": sha256(body_state_path),
        },
        "authorization": {
            "independent_test": not failures,
            "onnx_backend_gates": False,
            "frozen_video_replay": False,
            "deployment": False,
        },
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-state", type=Path, required=True)
    parser.add_argument("--body-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage106 evidence: {args.output}")
    result = decide(args.color_state.resolve(), args.body_state.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
