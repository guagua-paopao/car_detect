#!/usr/bin/env python3
"""Combine Stage109 color and Stage115 body validation evidence fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")


def color_gates(variant: dict[str, Any]) -> dict[str, bool]:
    static = variant["static"]
    track = variant["track_final"]
    stability = variant["stability"]
    return {
        "color_static_precision": static["precision"] >= 0.93,
        "color_static_coverage": static["coverage"] >= 0.25,
        "color_complex_unknown_reduction": variant["unknown_reduction"] >= 0.20,
        "color_track_precision": track["precision"] >= 0.93,
        "color_track_coverage": track["coverage"] >= 0.25,
        "color_track_stability": stability["transition_stability"] >= 0.95,
    }


def body_gates(variant: dict[str, Any]) -> dict[str, bool]:
    static = variant["static"]
    track = variant["track_final"]
    stability = variant["stability"]
    return {
        "body_static_precision": static["precision"] >= 0.93,
        "body_static_coverage": static["coverage"] >= 0.45,
        "body_complex_coverage_gain": variant["complex_coverage_gain"] >= 0.15,
        "body_track_precision": track["precision"] >= 0.93,
        "body_track_coverage": track["coverage"] >= 0.45,
        "body_track_stability": stability["transition_stability"] >= 0.95,
    }


def select_variant(
    state: dict[str, Any], gate_function, role: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    audited = []
    for variant in state.get("variants", []):
        gates = gate_function(variant)
        audited.append({**variant, "recomputed_gates": gates, "recomputed_all_pass": all(gates.values())})
    qualified = [variant for variant in audited if variant["recomputed_all_pass"]]
    if not qualified:
        return None, audited
    if role == "color":
        selected = max(
            qualified,
            key=lambda item: (
                item["unknown_reduction"], item["static"]["coverage"],
                item["track_final"]["coverage"], item["static"]["precision"],
            ),
        )
    else:
        selected = max(
            qualified,
            key=lambda item: (
                item["static"]["coverage"], item["track_final"]["coverage"],
                item["complex_coverage_gain"], item["static"]["precision"],
            ),
        )
    return selected, audited


def decide(color_state: dict[str, Any], body_state: dict[str, Any]) -> dict[str, Any]:
    for role, state in (("color", color_state), ("body", body_state)):
        if state.get("test_accessed") is not False or state.get("frozen_video_used") is not False:
            raise RuntimeError(f"{role} validation state lost test/frozen isolation")
        if state.get("production_model_modified") is not False:
            raise RuntimeError(f"{role} validation state modified production")
    selected_color, audited_colors = select_variant(color_state, color_gates, "color")
    selected_body, audited_bodies = select_variant(body_state, body_gates, "body")
    authorized = selected_color is not None and selected_body is not None
    return {
        "status": (
            "pass_independent_test_authorized"
            if authorized
            else "component_repair_required_fail_closed"
        ),
        "independent_test_authorized": authorized,
        "candidate": (
            {
                "body_variant": selected_body["variant"],
                "body_specialist_checkpoint": selected_body["specialist_checkpoint"],
                "body_specialist_checkpoint_sha256": selected_body["specialist_checkpoint_sha256"],
                "body_class_thresholds": selected_body["thresholds"],
                "body_report_sha256": selected_body["report_sha256"],
                "color_variant": selected_color["variant"],
                "color_class_thresholds": selected_color["thresholds"],
                "color_report_sha256": selected_color["report_sha256"],
            }
            if authorized
            else None
        ),
        "audit": {"color_variants": audited_colors, "body_variants": audited_bodies},
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-state", type=Path, required=True)
    parser.add_argument("--expected-color-state-sha256", required=True)
    parser.add_argument("--body-state", type=Path, required=True)
    parser.add_argument("--expected-body-state-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage116 state")
    assert_hash(args.color_state, args.expected_color_state_sha256, "color state")
    assert_hash(args.body_state, args.expected_body_state_sha256, "body state")
    color_state = json.loads(args.color_state.read_text(encoding="utf-8"))
    body_state = json.loads(args.body_state.read_text(encoding="utf-8"))
    output = decide(color_state, body_state)
    output["inputs"] = {
        "color_state": str(args.color_state.resolve()),
        "color_state_sha256": sha256(args.color_state),
        "body_state": str(args.body_state.resolve()),
        "body_state_sha256": sha256(args.body_state),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": output["status"], "candidate": output["candidate"]}, ensure_ascii=False))
    return 0 if output["independent_test_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
