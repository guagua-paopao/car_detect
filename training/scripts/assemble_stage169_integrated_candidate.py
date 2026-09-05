#!/usr/bin/env python3
"""Assemble a fail-closed body/color candidate after validation-only gates pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-state", type=Path, required=True)
    parser.add_argument("--color-state", type=Path, required=True)
    parser.add_argument("--expected-color-state-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--expected-production-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_sha(path: Path, expected: str, role: str) -> None:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: {actual}")


def validate_geometry_contract(
    body_checkpoint: dict[str, Any],
    color_checkpoint: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    body_size = int(body_checkpoint.get("input_size", 0))
    color_size = int(color_checkpoint.get("input_size", 0))
    body_mode = str(body_checkpoint.get("resize_mode", "stretch"))
    color_mode = str(color_checkpoint.get("resize_mode", "stretch"))
    if (body_size, color_size) != (256, 256):
        raise RuntimeError(f"integrated input size must be 256/256, got {body_size}/{color_size}")
    if (body_mode, color_mode) != ("stretch", "stretch"):
        raise RuntimeError(f"integrated resize mode must be stretch/stretch, got {body_mode}/{color_mode}")
    if body_checkpoint.get("body_types") != labels.get("body_types"):
        raise RuntimeError("body checkpoint taxonomy does not match vehicle labels v2")
    if color_checkpoint.get("color_types") != labels.get("colors"):
        raise RuntimeError("color checkpoint taxonomy does not match vehicle labels v2")
    return {
        "input_size": 256,
        "body_resize_mode": body_mode,
        "color_resize_mode": color_mode,
        "current_adapter_compatible": True,
    }


def next_stage_authorization(passed: bool) -> dict[str, bool]:
    """Authorize only the next safe operation after component validation.

    Stage169 may authorize construction and sealing of a fresh Stage160
    holdout.  It must not authorize reading that holdout for inference, backend
    export, frozen-video replay, or deployment.
    """
    return {
        "stage160_new_holdout_build": passed,
        "independent_test_inference": False,
        "onnx_backend_gates": False,
        "frozen_video_replay": False,
        "deployment": False,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise FileExistsError("refusing to overwrite Stage169 evidence")
    assert_sha(args.color_state, args.expected_color_state_sha256, "color validation state")
    assert_sha(args.labels, args.expected_labels_sha256, "labels")
    assert_sha(args.production_config, args.expected_production_config_sha256, "production configuration")

    body_state = json.loads(args.body_state.read_text(encoding="utf-8"))
    color_state = json.loads(args.color_state.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    production = json.loads(args.production_config.read_text(encoding="utf-8"))
    body = body_state.get("selected")
    color = color_state.get("selected_color")
    rejection_reasons = []
    if body_state.get("status") != "body_candidate_pass" or not body:
        rejection_reasons.append("Stage168 has no validation-qualified integration-compatible body candidate")
    if not color or color.get("qualified") is not True:
        rejection_reasons.append("Stage159 R8 has no validation-qualified color candidate")
    if production.get("models", {}).get("vehicle_attribute", {}).get("artifact") != "vehicle-attr-agent-e-224":
        raise RuntimeError("production baseline artifact changed")

    result: dict[str, Any] = {
        "schema_version": "stage169-integrated-candidate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "integrated_candidate_rejected_fail_closed",
        "rejection_reasons": rejection_reasons,
        "authorization": next_stage_authorization(False),
        "inputs": {
            "body_state": str(args.body_state.resolve()),
            "body_state_sha256": sha256(args.body_state),
            "color_state": str(args.color_state.resolve()),
            "color_state_sha256": sha256(args.color_state),
            "labels_sha256": sha256(args.labels),
            "production_config_sha256": sha256(args.production_config),
        },
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "next_action": "repair the rejected validation component without accessing test or frozen data",
    }

    if not rejection_reasons:
        import torch

        body_path = Path(body["candidate_checkpoint"])
        color_path = Path(color["checkpoint"])
        assert_sha(body_path, body["candidate_checkpoint_sha256"], "body checkpoint")
        assert_sha(color_path, color["checkpoint_sha256"], "color checkpoint")
        body_report_path = Path(body["report_path"])
        assert_sha(body_report_path, body["report_sha256"], "body validation report")
        body_report = json.loads(body_report_path.read_text(encoding="utf-8"))
        if not all(body_report.get("gates", {}).get("gates", {}).values()):
            raise RuntimeError("body report gates changed")
        if body_report["track_fusion"]["selection_search"]["base_thresholds_never_lowered"] is not True:
            raise RuntimeError("body short-track threshold policy changed")
        if not all(color.get("gates", {}).values()):
            raise RuntimeError("color validation gates changed")
        body_checkpoint = torch.load(body_path, map_location="cpu", weights_only=False)
        color_checkpoint = torch.load(color_path, map_location="cpu", weights_only=False)
        geometry = validate_geometry_contract(body_checkpoint, color_checkpoint, labels)
        result.update({
            "status": "integrated_candidate_pass_stage160_holdout_build_authorized",
            "rejection_reasons": [],
            "authorization": next_stage_authorization(True),
            "body": {
                "checkpoint": str(body_path),
                "checkpoint_sha256": sha256(body_path),
                "validation_report": str(body_report_path),
                "validation_report_sha256": sha256(body_report_path),
                "thresholds": body_report["selection"]["thresholds"],
                "overall": body["overall"],
                "complex": body["complex"],
                "comparison": body["comparison"],
                "track": body["track"],
            },
            "color": {
                "checkpoint": str(color_path),
                "checkpoint_sha256": sha256(color_path),
                "validation_report_sha256": color["report_sha256"],
                "thresholds": color["thresholds"],
                "overall": color["overall"],
                "complex": color["complex"],
                "comparison": color["comparison"],
                "track": color["track"],
            },
            "geometry_contract": geometry,
            "next_action": "build and SHA256-seal a fresh Stage160 holdout before any one-time test inference; keep backend gates locked",
        })

    args.output.parent.mkdir(parents=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "rejection_reasons": result["rejection_reasons"]}))
    return 0 if result["status"] == "integrated_candidate_pass_stage160_holdout_build_authorized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
