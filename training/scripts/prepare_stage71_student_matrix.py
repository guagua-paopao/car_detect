#!/usr/bin/env python3
"""Materialize a Stage71 student matrix only from a passing teacher screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested_number(item: dict, *keys: str) -> float:
    value = item
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pair_score(pair: dict) -> tuple[float, float, str]:
    hard = (
        nested_number(pair, "hard", "body_type", "coverage")
        + nested_number(pair, "hard", "color", "coverage")
    )
    track = (
        nested_number(pair, "ua_track", "body_type", "coverage")
        + nested_number(pair, "vfg7", "body", "track", "body_fusion5_final", "coverage")
        + nested_number(pair, "vfg7", "color", "track", "color_fusion5_final", "coverage")
    )
    return hard + track, hard, str(pair.get("pair_id", ""))


def select_passing_pair(report: dict) -> dict:
    if report.get("status") != "pass_pairs_available" or not report.get("passing_pairs"):
        raise RuntimeError("teacher screen has no passing pair")
    policy = report.get("policy", {})
    required = {
        "validation_only": True,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"teacher screen policy mismatch: {key}")
    passing_ids = set(report["passing_pairs"])
    candidates = [
        item for item in report.get("pairs", [])
        if item.get("pair_id") in passing_ids
        and item.get("screen_status") == "pass"
        and all(item.get("gates", {}).values())
        and item.get("gates")
    ]
    if not candidates:
        raise RuntimeError("teacher screen passing list is not backed by all-green gates")
    return max(candidates, key=pair_score)


def bind_teacher(candidate: dict, pair: dict) -> None:
    specialist = candidate.get("specialist")
    if specialist == "body":
        candidate["body_teacher_checkpoint"] = pair["body_checkpoint"]
        candidate["body_teacher_checkpoint_sha256"] = pair["body_checkpoint_sha256"]
    elif specialist == "color":
        candidate["color_teacher_checkpoint"] = pair["color_checkpoint"]
        candidate["color_teacher_checkpoint_sha256"] = pair["color_checkpoint_sha256"]
    else:
        raise RuntimeError(f"unsupported student specialist: {specialist}")


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--validation-state", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite student matrix: {args.output}")
    require_sha(args.template, args.expected_template_sha256, "student template")
    template = json.loads(args.template.read_text(encoding="utf-8"))
    if template.get("status") != "template_waiting_for_teacher_validation":
        raise RuntimeError("student template is not fail-closed")

    state = json.loads(args.validation_state.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError("teacher validation state is not complete")
    if Path(state.get("report", "")).resolve() != args.validation_report.resolve():
        raise RuntimeError("teacher validation state points to another report")
    require_sha(args.validation_report, state.get("report_sha256", ""), "teacher validation report")
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    pair = select_passing_pair(report)

    training_state = Path(report["training_state"])
    require_sha(training_state, report["training_state_sha256"], "teacher training state")
    training = json.loads(training_state.read_text(encoding="utf-8"))
    if training.get("status") != "complete_validation_only" or training.get("failed"):
        raise RuntimeError("teacher training did not complete validation-only")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if training.get(key) is not False:
            raise RuntimeError(f"teacher training policy mismatch: {key}")

    matrix = copy.deepcopy(template)
    matrix["status"] = "prepared_validation_only"
    matrix["created_at"] = datetime.now(timezone.utc).isoformat()
    matrix["teacher_pair_id"] = pair["pair_id"]
    matrix["source_template"] = {"path": str(args.template.resolve()), "sha256": sha256(args.template)}
    matrix["immutable_inputs"]["teacher_validation_report"] = {
        "path": str(args.validation_report.resolve()), "sha256": sha256(args.validation_report),
    }
    matrix["immutable_inputs"]["teacher_validation_state"] = {
        "path": str(args.validation_state.resolve()), "sha256": sha256(args.validation_state),
    }
    matrix["immutable_inputs"]["teacher_training_state"] = {
        "path": str(training_state.resolve()), "sha256": sha256(training_state),
    }
    for name, item in matrix["immutable_inputs"].items():
        require_sha(Path(item["path"]), item["sha256"], name)
    for candidate in matrix.get("candidates", []):
        bind_teacher(candidate, pair)
        for field in ("init_checkpoint", "body_teacher_checkpoint", "color_teacher_checkpoint"):
            if candidate.get(field):
                require_sha(Path(candidate[field]), candidate[f"{field}_sha256"], f"{candidate['candidate_id']} {field}")
    if len(matrix.get("candidates", [])) != 5:
        raise RuntimeError("Stage71 student candidate matrix is incomplete")
    matrix["materialization_policy"] = {
        "teacher_pair_selected_only_from_all_green_validation_pairs": True,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "research_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, matrix)
    print(json.dumps({
        "status": matrix["status"],
        "teacher_pair_id": pair["pair_id"],
        "candidates": len(matrix["candidates"]),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "test_accessed": False,
        "frozen_video_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
