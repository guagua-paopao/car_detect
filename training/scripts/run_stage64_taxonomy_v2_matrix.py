#!/usr/bin/env python3
"""Run isolated validation-only taxonomy-v2 candidates after the v1 queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


META_KEYS = {"candidate_id", "purpose", "init_checkpoint_sha256"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_arguments(parameters: dict) -> list[str]:
    result: list[str] = []
    for key, value in parameters.items():
        if key in META_KEYS or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def require_false(payload: dict, keys: tuple[str, ...], name: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise RuntimeError(f"{name} policy violation: {key} is not false")


def validate_unlabeled_evidence(matrix: dict) -> tuple[Path, Path]:
    manifest = Path(matrix["unlabeled_manifest"])
    report_path = Path(matrix["unlabeled_report"])
    for path in (manifest, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(manifest).upper() != matrix["unlabeled_manifest_sha256"].upper():
        raise RuntimeError("unlabeled adverse-scene manifest pinned hash mismatch")
    if sha256(report_path).upper() != matrix["unlabeled_report_sha256"].upper():
        raise RuntimeError("unlabeled adverse-scene report pinned hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("unlabeled adverse-scene report did not pass")
    if report.get("output_manifest_sha256", "").lower() != sha256(manifest):
        raise RuntimeError("unlabeled adverse-scene manifest hash mismatch")
    policy = report.get("policy", {})
    required_true = (
        "all_inputs_hash_verified",
        "all_rows_train_only",
        "all_attributes_remain_unknown_unsupervised",
        "duplicate_image_paths_rejected",
    )
    for key in required_true:
        if policy.get(key) is not True:
            raise RuntimeError(f"unlabeled adverse-scene policy violation: {key} is not true")
    require_false(
        policy,
        ("validation_or_test_used", "frozen_video_used", "production_model_modified", "deployment_performed"),
        "unlabeled adverse-scene report",
    )
    return manifest, report_path


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
        raise FileExistsError("refusing to overwrite Stage64 evidence")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()).upper() != matrix.get("runner_script_sha256", "").upper():
        raise RuntimeError("Stage64 runner script hash mismatch")
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage64 matrix is not prepared")
    execution = matrix.get("execution_policy", {})
    require_false(
        execution,
        ("iterative_test_access", "frozen_video_used", "production_model_modified", "deployment_performed"),
        "matrix",
    )
    if execution.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage64 matrix lost mandatory --skip-test")
    if execution.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage64 matrix lost the deployment pause")
    if execution.get("license_policy") != "offline_research_only_no_deployment":
        raise RuntimeError("Stage64 matrix license policy is not fail closed")
    if execution.get("exact_body_gate_not_replaced_by_family_metric") is not True:
        raise RuntimeError("Stage64 matrix may substitute a family metric for the exact gate")
    for raw_path, expected_hash in matrix.get("source_hashes", {}).items():
        source_path = Path(raw_path)
        if not source_path.is_file() or sha256(source_path).upper() != expected_hash.upper():
            raise RuntimeError(f"Stage64 source hash mismatch: {source_path}")

    training_script = Path(matrix["training_script"])
    labels = Path(matrix["labels"])
    manifest = Path(matrix["manifest"])
    manifest_report_path = Path(matrix["manifest_report"])
    for path in (training_script, labels, manifest, manifest_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(training_script).upper() != matrix["training_script_sha256"].upper():
        raise RuntimeError("taxonomy-v2 training script hash mismatch")
    if sha256(labels).upper() != matrix["labels_sha256"].upper():
        raise RuntimeError("taxonomy-v2 labels hash mismatch")
    if sha256(manifest).upper() != matrix["manifest_sha256"].upper():
        raise RuntimeError("taxonomy-v2 manifest pinned hash mismatch")
    if sha256(manifest_report_path).upper() != matrix["manifest_report_sha256"].upper():
        raise RuntimeError("taxonomy-v2 manifest report pinned hash mismatch")
    labels_payload = json.loads(labels.read_text(encoding="utf-8"))
    if labels_payload.get("deployment_status") != "offline_candidate_only":
        raise RuntimeError("taxonomy-v2 labels are not isolated offline labels")

    report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("taxonomy-v2 manifest report did not pass")
    if report.get("output_manifest_sha256", "").lower() != sha256(manifest):
        raise RuntimeError("taxonomy-v2 manifest hash mismatch")
    policy = report.get("policy", {})
    if policy.get("frozen_video_used") is not False or policy.get("production_contract_modified") is not False:
        raise RuntimeError("taxonomy-v2 manifest violated isolation policy")
    if policy.get("validation_and_test_membership_preserved_exactly") is not True:
        raise RuntimeError("taxonomy-v2 manifest changed held-out membership")

    unlabeled_manifest, unlabeled_report_path = validate_unlabeled_evidence(matrix)

    candidates = list(matrix.get("candidates", []))
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if not candidates or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("Stage64 candidate IDs are empty or duplicated")
    for candidate in candidates:
        if candidate.get("body_hierarchy", matrix["common"].get("body_hierarchy")) != "truck_family":
            raise RuntimeError(f"{candidate['candidate_id']} lost truck-family hierarchy")
        init = candidate.get("init_checkpoint")
        if init:
            init_path = Path(init)
            if not init_path.is_file():
                raise FileNotFoundError(init)
            if sha256(init_path).upper() != candidate.get("init_checkpoint_sha256", "").upper():
                raise RuntimeError(f"{candidate['candidate_id']} initialization checkpoint hash mismatch")
        if candidate.get("unlabeled_consistency_weight", 0) > 0:
            if candidate.get("unlabeled_consistency_mode") != "feature":
                raise RuntimeError("Stage64 adverse consistency must use feature mode")
            if Path(candidate["unlabeled_manifest"]) != unlabeled_manifest:
                raise RuntimeError("Stage64 candidate references an unapproved unlabeled manifest")

    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only",
            "matrix_sha256": sha256(args.matrix),
            "manifest_sha256": sha256(manifest),
            "manifest_report_sha256": sha256(manifest_report_path),
            "labels_sha256": sha256(labels),
            "unlabeled_manifest_sha256": sha256(unlabeled_manifest),
            "unlabeled_report_sha256": sha256(unlabeled_report_path),
            "candidate_count": len(candidates),
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        }, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "stage64-taxonomy-v2-run-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "manifest_report_sha256": sha256(manifest_report_path),
        "labels_sha256": sha256(labels),
        "training_script_sha256": sha256(training_script),
        "unlabeled_manifest": str(unlabeled_manifest.resolve()),
        "unlabeled_manifest_sha256": sha256(unlabeled_manifest),
        "unlabeled_report_sha256": sha256(unlabeled_report_path),
        "candidates_total": len(candidates),
        "completed": [],
        "failed": [],
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "license_policy": execution["license_policy"],
    }
    atomic_json(args.state, state)
    common = {
        **matrix["common"],
        "manifest": str(manifest.resolve()),
        "labels": str(labels.resolve()),
        "device": args.device,
        "dataset_version": "attribute-domain-v2-stage64-taxonomy-v2",
        "code_revision": args.code_revision,
    }
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        parameters = {**common, **candidate, "output_dir": str(output_dir)}
        command = [sys.executable, str(training_script), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "started_at": now(), "command": command}
        state["active"] = active
        state["updated_at"] = now()
        atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(
                command,
                cwd=str(training_script.resolve().parents[2]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        best = output_dir / "best.pt"
        record = {
            **active,
            "finished_at": now(),
            "return_code": result.returncode,
            "stdout_log": str(log.resolve()),
            "stdout_log_sha256": sha256(log),
            "best_checkpoint": str(best.resolve()) if best.is_file() else None,
            "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
        }
        state["completed" if result.returncode == 0 and best.is_file() else "failed"].append(record)
        state.pop("active", None)
        state["updated_at"] = now()
        atomic_json(args.state, state)
    state["status"] = "complete_validation_only"
    state["updated_at"] = now()
    atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
