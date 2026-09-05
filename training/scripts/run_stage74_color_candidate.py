#!/usr/bin/env python3
"""Run the fail-closed Stage74 CCTV-joint color specialist candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


META_KEYS = {"candidate_id", "purpose", "specialist", "init_checkpoint_sha256"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_arguments(parameters: dict) -> list[str]:
    arguments: list[str] = []
    for key, value in parameters.items():
        if key in META_KEYS or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        else:
            arguments.extend([flag, str(value)])
    return arguments


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usable_color_train_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and row.get("color") not in {"", "unknown", None}
    ]


def expected_supervised_quota(rows: list[dict[str, str]], parameters: dict) -> dict[str, float]:
    weighted: list[tuple[float, bool, bool, bool, bool]] = []
    for row in rows:
        try:
            hard_score = max(0.0, float(row.get("hard_score", "0") or 0.0))
        except ValueError:
            hard_score = 0.0
        weight = 1.0 + float(parameters.get("hard_sample_weight", 0.0)) * min(hard_score, 8.0) / 8.0
        try:
            explicit = float(row.get("sample_weight", "1") or 1.0)
        except ValueError:
            explicit = 1.0
        weight *= max(0.10, min(10.0, explicit))
        small = truthy(row.get("small_target")) or row.get("vehicle_size") == "small"
        night = truthy(row.get("night"))
        occluded = truthy(row.get("occluded")) or truthy(row.get("truncated"))
        cctv = row.get("stage71_origin") == "cctv"
        if small:
            weight *= 1.0 + float(parameters.get("small_sample_weight", 0.0))
        if night:
            weight *= float(parameters.get("night_sample_weight", 1.0))
        if occluded:
            weight *= float(parameters.get("occlusion_sample_weight", 1.0))
        weighted.append((weight, night, small, occluded, cctv))
    total = sum(item[0] for item in weighted)
    if total <= 0:
        raise RuntimeError("supervised sampler has zero total weight")
    return {
        "rows": len(rows),
        "night": sum(item[0] for item in weighted if item[1]) / total,
        "small": sum(item[0] for item in weighted if item[2]) / total,
        "occluded_or_truncated": sum(item[0] for item in weighted if item[3]) / total,
        "cctv": sum(item[0] for item in weighted if item[4]) / total,
    }


def expected_unlabeled_night(rows: list[dict[str, str]], night_weight: float) -> float:
    train = [row for row in rows if row.get("split") == "train"]
    if not train or any(
        row.get("body_type") not in {"", "unknown", None}
        or row.get("color") not in {"", "unknown", None}
        or truthy(row.get("body_type_supervised"))
        or truthy(row.get("color_supervised"))
        for row in train
    ):
        raise RuntimeError("unlabeled stream is empty or contains a class target")
    night = sum(truthy(row.get("night")) for row in train)
    return (night * night_weight) / (night * night_weight + len(train) - night)


def verify_report(report: dict, manifest_sha256: str, matrix: dict) -> None:
    if report.get("status") != "pass_joint_research_manifest_available":
        raise RuntimeError("Stage74 joint report did not pass")
    if report.get("eligibility") != "research-only_non-deployable":
        raise RuntimeError("Stage74 joint report lost research-only isolation")
    if report.get("output_manifest_sha256", "").lower() != manifest_sha256.lower():
        raise RuntimeError("Stage74 report does not bind the color manifest")
    if report.get("joint_train_rows", 0) < int(matrix["quota_policy"]["minimum_color_train_rows"]):
        raise RuntimeError("Stage74 color train rows are below the fixed minimum")
    if report.get("retained_new_rows", 0) < int(matrix["quota_policy"]["minimum_new_cctv_rows"]):
        raise RuntimeError("Stage74 retained CCTV rows are below the fixed minimum")
    if report.get("supported_new_classes", 0) < int(matrix["quota_policy"]["minimum_supported_new_classes"]):
        raise RuntimeError("Stage74 retained CCTV class support is below the fixed minimum")
    if report.get("validation_projection_sha256_before") != report.get("validation_projection_sha256_after"):
        raise RuntimeError("Stage74 validation projection changed")
    if report.get("gates", {}).get("group_leaks") != 0:
        raise RuntimeError("Stage74 joint report retains a group leak")
    policy = report.get("policy", {})
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage74 report policy mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage74 candidate evidence")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage74 matrix is not prepared")
    if matrix.get("runner_sha256", "").lower() != sha256(Path(__file__).resolve()):
        raise RuntimeError("Stage74 runner SHA256 mismatch")
    policy = matrix.get("execution_policy", {})
    required_policy = {
        "iterative_test_access": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "research_only": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"Stage74 execution policy mismatch: {key}")
    if policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage74 lost test isolation")

    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"Stage74 immutable input mismatch: {name}")
        inputs[name] = path
    needed = {"training_script", "labels", "manifest_report", "color_manifest", "unlabeled_manifest", "init_checkpoint"}
    if not needed.issubset(inputs):
        raise RuntimeError("Stage74 immutable input set is incomplete")

    manifest_sha = sha256(inputs["color_manifest"])
    report = json.loads(inputs["manifest_report"].read_text(encoding="utf-8"))
    verify_report(report, manifest_sha, matrix)

    safety_root = Path(matrix.get("datasets_safety_root", "")).resolve()
    if not safety_root.is_dir():
        raise RuntimeError("Stage74 datasets safety root is missing")
    color_rows = load_rows(inputs["color_manifest"])
    unlabeled_rows = load_rows(inputs["unlabeled_manifest"])
    for name, manifest, rows in (
        ("color", inputs["color_manifest"], color_rows),
        ("unlabeled", inputs["unlabeled_manifest"], unlabeled_rows),
    ):
        for row in rows:
            raw = Path(row.get("image_path", ""))
            resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
            try:
                resolved.relative_to(safety_root)
            except ValueError as error:
                raise RuntimeError(f"{name} image escapes the datasets safety root: {resolved}") from error
            if row.get("split") == "test":
                raise RuntimeError(f"{name} manifest contains a test row")

    candidate = matrix.get("candidate", {})
    parameters = {**matrix.get("common", {}), **candidate}
    if not (
        candidate.get("specialist") == "color"
        and parameters.get("architecture") == "convnext_tiny"
        and parameters.get("selection_head") == "color"
        and float(parameters.get("color_loss_weight", 0)) > 0
        and float(parameters.get("body_loss_weight", -1)) == 0
        and float(parameters.get("unlabeled_color_weight", 0)) > 0
        and float(parameters.get("unlabeled_body_weight", -1)) == 0
        and float(parameters.get("pseudo_label_weight", 1)) < 1
    ):
        raise RuntimeError("Stage74 candidate lost color-specialist or conservative pseudo-label isolation")
    if Path(candidate.get("init_checkpoint", "")).resolve() != inputs["init_checkpoint"].resolve():
        raise RuntimeError("Stage74 candidate init checkpoint path mismatch")
    if candidate.get("init_checkpoint_sha256", "").lower() != sha256(inputs["init_checkpoint"]):
        raise RuntimeError("Stage74 candidate init checkpoint SHA256 mismatch")

    usable = usable_color_train_rows(color_rows)
    if len(usable) != int(report.get("joint_train_rows", -1)):
        raise RuntimeError("Stage74 usable color row count does not match the joint report")
    supervised = expected_supervised_quota(usable, parameters)
    unlabeled_night = expected_unlabeled_night(
        unlabeled_rows,
        float(parameters.get("unlabeled_night_sample_weight", 1.0)),
    )
    quotas = matrix["quota_policy"]
    required_fractions = {
        "cctv": float(quotas["minimum_expected_cctv_fraction"]),
        "night": float(quotas["minimum_expected_supervised_night_fraction"]),
        "small": float(quotas["minimum_expected_small_fraction"]),
    }
    for key, minimum in required_fractions.items():
        if supervised[key] < minimum:
            raise RuntimeError(f"Stage74 expected supervised {key} quota failed")
    if unlabeled_night < float(quotas["minimum_expected_unlabeled_night_fraction"]):
        raise RuntimeError("Stage74 expected unlabeled night quota failed")
    quota_report = {"supervised": supervised, "unlabeled_night": unlabeled_night}
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only",
            "expected_sampler_quotas": quota_report,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        }, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": "stage74-color-candidate-run-v1",
        "status": "running_validation_only",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "manifest_sha256": manifest_sha,
        "expected_sampler_quotas": quota_report,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "eligibility": "research-only_non-deployable",
    }
    atomic_json(args.state, state)
    output_dir = args.output_root / candidate["candidate_id"]
    run_parameters = {
        **parameters,
        "manifest": str(inputs["color_manifest"].resolve()),
        "unlabeled_manifest": str(inputs["unlabeled_manifest"].resolve()),
        "unlabeled_root": str(safety_root),
        "labels": str(inputs["labels"].resolve()),
        "device": args.device,
        "dataset_version": "attribute-domain-v2-stage74-cctv-joint-color-research-v1",
        "code_revision": args.code_revision,
        "output_dir": str(output_dir),
    }
    command = [sys.executable, str(inputs["training_script"]), *command_arguments(run_parameters)]
    if "--skip-test" not in command:
        raise RuntimeError("Stage74 candidate lost --skip-test")
    state["active"] = {"candidate_id": candidate["candidate_id"], "started_at": now(), "command": command}
    state["updated_at"] = now()
    atomic_json(args.state, state)
    log = args.output_root / f"{candidate['candidate_id']}.stdout.log"
    with log.open("wb") as handle:
        result = subprocess.run(
            command,
            cwd=str(inputs["training_script"].resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    best = output_dir / "best.pt"
    state["result"] = {
        "candidate_id": candidate["candidate_id"],
        "finished_at": now(),
        "return_code": result.returncode,
        "stdout_log": str(log.resolve()),
        "stdout_log_sha256": sha256(log),
        "best_checkpoint": str(best.resolve()) if best.is_file() else None,
        "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
    }
    state.pop("active", None)
    state["status"] = "complete_validation_only" if result.returncode == 0 and best.is_file() else "failed_closed"
    state["updated_at"] = now()
    atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "return_code": result.returncode}))
    return 0 if state["status"] == "complete_validation_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
