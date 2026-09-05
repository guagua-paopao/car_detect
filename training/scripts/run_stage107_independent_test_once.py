#!/usr/bin/env python3
"""Run the one-time Stage107 hard/VFG/UA independent test chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_candidate_inputs(
    stage106_state: Path,
    expected_stage106_sha256: str,
    stage104_dir: Path,
    stage105_dir: Path,
) -> dict[str, Any]:
    require_sha(stage106_state, expected_stage106_sha256, "Stage106 state")
    gate = read_json(stage106_state)
    if gate.get("status") != "ready_for_integrated_independent_test":
        raise RuntimeError("Stage106 did not pass the integrated component gate")
    if gate.get("authorization", {}).get("independent_test") is not True:
        raise RuntimeError("Stage106 did not authorize independent test")
    color = gate.get("selected_color")
    body = gate.get("selected_body")
    if not isinstance(color, dict) or not isinstance(body, dict):
        raise RuntimeError("Stage106 selected components are missing")
    color_report = stage104_dir / f"{color['variant']}.json"
    body_report = stage105_dir / f"{body['variant']}.json"
    require_sha(color_report, str(color["report_sha256"]), "Stage104 selected report")
    require_sha(body_report, str(body["report_sha256"]), "Stage105 selected report")
    color_evidence = read_json(color_report)
    body_evidence = read_json(body_report)
    for name, report in (("color", color_evidence), ("body", body_evidence)):
        if report.get("status") != "complete_validation_only":
            raise RuntimeError(f"selected {name} report is not validation-only complete")
        policy = report.get("policy", {})
        if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
            raise RuntimeError(f"selected {name} report has unsafe policy")
    body_threshold = float(body["static"]["threshold"])
    color_threshold = float(color["static"]["threshold"])
    subtype_threshold = float(body["subtype_threshold"])
    if abs(float(body_evidence["threshold_selection"]["body_exact"]["threshold"]) - body_threshold) > 1e-12:
        raise RuntimeError("Stage106 body threshold does not match selected Stage105 report")
    if abs(float(color_evidence["threshold_selection"]["color_shared"]["threshold"]) - color_threshold) > 1e-12:
        raise RuntimeError("Stage106 color threshold does not match selected Stage104 report")
    if abs(float(body_evidence["inputs"]["body_specialist_subtype_threshold"]) - subtype_threshold) > 1e-12:
        raise RuntimeError("Stage106 subtype threshold does not match selected Stage105 report")
    body_inputs = body_evidence["inputs"]
    color_inputs = color_evidence["inputs"]
    result = {
        "body_checkpoint": Path(body_inputs["body_checkpoint"]),
        "body_checkpoint_sha256": str(body_inputs["body_checkpoint_sha256"]),
        "body_specialist_checkpoint": Path(body_inputs["body_specialist_checkpoint"]),
        "body_specialist_checkpoint_sha256": str(body_inputs["body_specialist_checkpoint_sha256"]),
        "color_checkpoint": Path(color_inputs["color_checkpoint"]),
        "color_checkpoint_sha256": str(color_inputs["color_checkpoint_sha256"]),
        "body_threshold": body_threshold,
        "color_threshold": color_threshold,
        "body_specialist_subtype_threshold": subtype_threshold,
        "selected_body_variant": body["variant"],
        "selected_color_variant": color["variant"],
        "stage104_report": color_report,
        "stage104_report_sha256": sha256(color_report),
        "stage105_report": body_report,
        "stage105_report_sha256": sha256(body_report),
    }
    for key in ("body_checkpoint", "body_specialist_checkpoint", "color_checkpoint"):
        require_sha(result[key], result[f"{key}_sha256"], key.replace("_", " "))
    return result


def extract_stage147_candidate_inputs(
    gate_path: Path,
    expected_gate_sha256: str,
    color_reports: Path,
    body_reports: Path,
    *,
    body_checkpoint: Path,
    expected_body_checkpoint_sha256: str,
    body_specialist_checkpoint: Path,
    expected_body_specialist_checkpoint_sha256: str,
    color_checkpoint: Path,
    expected_color_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Resolve a per-class-threshold Stage147 authorization before test access."""
    require_sha(gate_path, expected_gate_sha256, "Stage147 state")
    gate = read_json(gate_path)
    if gate.get("status") != "pass_independent_test_authorized":
        raise RuntimeError("Stage147 did not pass the integrated component gate")
    if gate.get("independent_test_authorized") is not True:
        raise RuntimeError("Stage147 did not authorize independent test")
    for key in (
        "test_accessed", "frozen_video_used", "production_model_modified",
        "backend_gates_run", "deployment_performed",
    ):
        if gate.get(key) is not False:
            raise RuntimeError(f"unsafe Stage147 state: {key}")
    candidate = gate.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("Stage147 selected candidate is missing")
    body_report = body_reports / str(candidate["body_variant"]) / "report.json"
    color_report = color_reports / str(candidate["color_variant"]) / "report.json"
    require_sha(body_report, str(candidate["body_report_sha256"]), "selected body report")
    require_sha(color_report, str(candidate["color_report_sha256"]), "selected color report")
    body_evidence, color_evidence = read_json(body_report), read_json(color_report)
    for name, report in (("body", body_evidence), ("color", color_evidence)):
        if report.get("status") != "complete_validation_only":
            raise RuntimeError(f"selected {name} report is not validation-only complete")
        policy = report.get("policy", {})
        if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
            raise RuntimeError(f"selected {name} report has unsafe policy")
    if body_evidence["selection"]["thresholds"] != candidate["body_class_thresholds"]:
        raise RuntimeError("Stage147 body thresholds do not match selected report")
    if color_evidence["selection"]["thresholds"] != candidate["color_class_thresholds"]:
        raise RuntimeError("Stage147 color thresholds do not match selected report")
    if float(body_evidence["inputs"]["body_specialist_subtype_threshold"]) != 0.50:
        raise RuntimeError("Stage147 specialist routing threshold is not the fixed 0.50 contract")
    pins = (
        (body_checkpoint, expected_body_checkpoint_sha256, "body checkpoint"),
        (
            body_specialist_checkpoint,
            expected_body_specialist_checkpoint_sha256,
            "body specialist checkpoint",
        ),
        (color_checkpoint, expected_color_checkpoint_sha256, "color checkpoint"),
    )
    for path, expected, role in pins:
        require_sha(path, expected, role)
    if str(candidate["body_specialist_checkpoint_sha256"]).lower() != expected_body_specialist_checkpoint_sha256.lower():
        raise RuntimeError("Stage147 specialist SHA256 does not match the pinned checkpoint")
    return {
        "body_checkpoint": body_checkpoint,
        "body_checkpoint_sha256": expected_body_checkpoint_sha256,
        "body_specialist_checkpoint": body_specialist_checkpoint,
        "body_specialist_checkpoint_sha256": expected_body_specialist_checkpoint_sha256,
        "color_checkpoint": color_checkpoint,
        "color_checkpoint_sha256": expected_color_checkpoint_sha256,
        "body_class_thresholds": candidate["body_class_thresholds"],
        "color_class_thresholds": candidate["color_class_thresholds"],
        "body_specialist_subtype_threshold": 0.50,
        "selected_body_variant": candidate["body_variant"],
        "selected_color_variant": candidate["color_variant"],
        "body_report": body_report,
        "body_report_sha256": sha256(body_report),
        "color_report": color_report,
        "color_report_sha256": sha256(color_report),
    }


def composite_gates(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    hard = reports["hard"]
    vfg = reports["vfg"]
    ua = reports["ua"]
    hard_gates = hard["gates"]
    vfg_gates = vfg["gates"]
    ua_gates = ua["gates"]
    result = {
        f"hard_{key}": bool(hard_gates[key])
        for key in (
            "body_static_precision",
            "body_static_coverage",
            "body_complex_coverage_gain",
            "color_static_precision",
            "color_static_coverage",
            "color_complex_unknown_reduction",
        )
    }
    result.update({f"vfg_{key}": bool(value) for key, value in vfg_gates.items()})
    result.update(
        {
            f"ua_{key}": bool(ua_gates[key])
            for key in (
                "body_track_precision",
                "body_track_coverage",
                "body_track_stability",
            )
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage106-state", type=Path, required=True)
    parser.add_argument("--expected-stage106-state-sha256", required=True)
    parser.add_argument("--stage104-dir", type=Path, required=True)
    parser.add_argument("--stage105-dir", type=Path, required=True)
    parser.add_argument("--test-view-builder", type=Path, required=True)
    parser.add_argument("--expected-test-view-builder-sha256", required=True)
    parser.add_argument("--fixed-test-evaluator", type=Path, required=True)
    parser.add_argument("--expected-fixed-test-evaluator-sha256", required=True)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--expected-base-evaluator-sha256", required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--expected-hard-manifest-sha256", required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--vfg-manifest", type=Path, required=True)
    parser.add_argument("--expected-vfg-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path)
    parser.add_argument("--expected-body-checkpoint-sha256")
    parser.add_argument("--body-specialist-checkpoint", type=Path)
    parser.add_argument("--expected-body-specialist-checkpoint-sha256")
    parser.add_argument("--color-checkpoint", type=Path)
    parser.add_argument("--expected-color-checkpoint-sha256")
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage107 one-time test evidence")
    code_pins = (
        (args.test_view_builder, args.expected_test_view_builder_sha256, "test-view builder"),
        (args.fixed_test_evaluator, args.expected_fixed_test_evaluator_sha256, "fixed test evaluator"),
        (args.base_evaluator, args.expected_base_evaluator_sha256, "base evaluator"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256, "baseline checkpoint"),
    )
    source_pins = (
        (args.hard_manifest, args.expected_hard_manifest_sha256, "hard manifest"),
        (args.ua_manifest, args.expected_ua_manifest_sha256, "UA manifest"),
        (args.vfg_manifest, args.expected_vfg_manifest_sha256, "VFG manifest"),
    )
    for path, expected, label in (*code_pins, *source_pins):
        require_sha(path, expected, label)
    gate = read_json(args.stage106_state)
    if gate.get("status") == "pass_independent_test_authorized":
        required = (
            args.body_checkpoint,
            args.expected_body_checkpoint_sha256,
            args.body_specialist_checkpoint,
            args.expected_body_specialist_checkpoint_sha256,
            args.color_checkpoint,
            args.expected_color_checkpoint_sha256,
        )
        if any(value is None for value in required):
            raise RuntimeError("Stage147 mode requires explicit pinned candidate checkpoints")
        candidate = extract_stage147_candidate_inputs(
            args.stage106_state,
            args.expected_stage106_state_sha256,
            args.stage104_dir,
            args.stage105_dir,
            body_checkpoint=args.body_checkpoint,
            expected_body_checkpoint_sha256=args.expected_body_checkpoint_sha256,
            body_specialist_checkpoint=args.body_specialist_checkpoint,
            expected_body_specialist_checkpoint_sha256=args.expected_body_specialist_checkpoint_sha256,
            color_checkpoint=args.color_checkpoint,
            expected_color_checkpoint_sha256=args.expected_color_checkpoint_sha256,
        )
    else:
        candidate = extract_candidate_inputs(
            args.stage106_state,
            args.expected_stage106_state_sha256,
            args.stage104_dir,
            args.stage105_dir,
        )
    args.output_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "stage107-independent-test-once-state-v1",
        "status": "authorized_not_yet_accessed",
        "created_at": now(),
        "updated_at": now(),
        "test_accessed": False,
        "test_used_for_selection": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    atomic_json(args.state, state)
    try:
        views_root = args.output_root / "test-views"
        state.update({"status": "building_test_views_once", "updated_at": now(), "test_accessed": True})
        atomic_json(args.state, state)
        run(
            [
                args.python,
                str(args.test_view_builder),
                "--stage106-state", str(args.stage106_state),
                "--expected-stage106-state-sha256", args.expected_stage106_state_sha256,
                "--hard-manifest", str(args.hard_manifest),
                "--expected-hard-manifest-sha256", args.expected_hard_manifest_sha256,
                "--ua-manifest", str(args.ua_manifest),
                "--expected-ua-manifest-sha256", args.expected_ua_manifest_sha256,
                "--vfg-manifest", str(args.vfg_manifest),
                "--expected-vfg-manifest-sha256", args.expected_vfg_manifest_sha256,
                "--datasets-safety-root", str(args.datasets_safety_root),
                "--output-root", str(views_root),
            ],
            args.output_root / "build-test-views.log",
        )
        view_report_path = views_root / "stage107-test-views-report.json"
        view_report = read_json(view_report_path)
        if view_report.get("status") != "pass_test_views_ready":
            raise RuntimeError("Stage107 test views did not pass")
        reports: dict[str, dict[str, Any]] = {}
        for name in ("hard", "vfg", "ua"):
            manifest = views_root / f"{name}.test.csv"
            output = args.output_root / f"{name}.test.report.json"
            run(
                [
                    args.python,
                    str(args.fixed_test_evaluator),
                    "--stage106-state", str(args.stage106_state),
                    "--expected-stage106-state-sha256", args.expected_stage106_state_sha256,
                    "--manifest", str(manifest),
                    "--expected-manifest-sha256", sha256(manifest),
                    "--labels", str(args.labels),
                    "--expected-labels-sha256", args.expected_labels_sha256,
                    "--body-checkpoint", str(candidate["body_checkpoint"]),
                    "--expected-body-checkpoint-sha256", candidate["body_checkpoint_sha256"],
                    "--body-specialist-checkpoint", str(candidate["body_specialist_checkpoint"]),
                    "--expected-body-specialist-checkpoint-sha256", candidate["body_specialist_checkpoint_sha256"],
                    "--color-checkpoint", str(candidate["color_checkpoint"]),
                    "--expected-color-checkpoint-sha256", candidate["color_checkpoint_sha256"],
                    "--baseline-checkpoint", str(args.baseline_checkpoint),
                    "--expected-baseline-checkpoint-sha256", args.expected_baseline_checkpoint_sha256,
                    "--datasets-safety-root", str(args.datasets_safety_root),
                    "--output", str(output),
                    "--device", args.device,
                    "--batch-size", str(args.batch_size),
                    "--workers", str(args.workers),
                ],
                args.output_root / f"{name}.test.log",
            )
            reports[name] = read_json(output)
            if reports[name].get("status") != "complete_test_once":
                raise RuntimeError(f"{name} test report incomplete")
        gates = composite_gates(reports)
        passed = all(gates.values())
        report = {
            "schema_version": "stage107-independent-test-once-report-v1",
            "created_at": now(),
            "status": "pass_backend_eligible" if passed else "fail_closed_before_backend",
            "candidate": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in candidate.items()
            },
            "test_views": {
                "report": str(view_report_path.resolve()),
                "report_sha256": sha256(view_report_path),
            },
            "dataset_reports": {
                name: {
                    "path": str((args.output_root / f"{name}.test.report.json").resolve()),
                    "sha256": sha256(args.output_root / f"{name}.test.report.json"),
                    "decision": reports[name]["decision"],
                }
                for name in reports
            },
            "composite_gates": gates,
            "all_composite_gates_pass": passed,
            "policy": {
                "test_accessed": True,
                "test_used_for_selection": False,
                "threshold_or_temperature_search": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "backend_gates_run": False,
                "deployment_performed": False,
            },
            "next_action": (
                "run isolated ONNX/TensorRT/C++ backend gates"
                if passed
                else "reject candidate and repair validation-identified component without reusing test for selection"
            ),
        }
        report_path = args.output_root / "stage107-independent-test-report.json"
        atomic_json(report_path, report)
        state.update(
            {
                "status": report["status"],
                "updated_at": now(),
                "report": str(report_path.resolve()),
                "report_sha256": sha256(report_path),
                "backend_eligible": passed,
            }
        )
        atomic_json(args.state, state)
        print(json.dumps({"status": report["status"], "gates": gates}, indent=2))
        return 0
    except Exception as error:
        state.update(
            {
                "status": "failed_closed_runtime",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "backend_eligible": False,
            }
        )
        atomic_json(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
