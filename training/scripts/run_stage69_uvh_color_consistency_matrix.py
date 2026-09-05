#!/usr/bin/env python3
"""Run the isolated Stage69 UVH/type plus real-domain consistency matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


META_KEYS = {"candidate_id", "purpose", "init_checkpoint_sha256", "teacher_checkpoint_sha256"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
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


def validate_matrix(matrix: dict, matrix_path: Path, runner_path: Path) -> dict[str, Path]:
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage69 matrix is not approved for validation-only training")
    if matrix.get("runner_sha256", "").lower() != sha256(runner_path):
        raise RuntimeError("Stage69 runner hash mismatch")
    policy = matrix.get("execution_policy", {})
    required_false = (
        "iterative_test_access", "frozen_video_used", "production_model_modified",
        "deployment_performed", "veRi_used", "apache_kaggle_used"
    )
    for key in required_false:
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage69 execution policy violation: {key}")
    if policy.get("required_flag") != "--skip-test" or policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage69 lost test isolation or deployment pause")
    paths: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual.lower() != item["sha256"].lower():
            raise RuntimeError(f"Stage69 immutable input mismatch: {name}")
        paths[name] = path
    required = {
        "training_script", "labels", "supervised_manifest", "unlabeled_manifest",
        "unlabeled_report", "stage67_builder_report", "stage67_near_report",
        "stage67_strata_report", "stage67_leakage_report", "stage67_license_evidence",
        "stage68_license_evidence", "stage68_finalization_report"
    }
    if not required.issubset(paths):
        raise RuntimeError("Stage69 immutable input set is incomplete")
    unlabeled = json.loads(paths["unlabeled_report"].read_text(encoding="utf-8"))
    if (
        unlabeled.get("status") != "pass"
        or unlabeled.get("output_sha256", "").lower() != sha256(paths["unlabeled_manifest"])
        or unlabeled.get("invalid_rows") != 0
        or unlabeled.get("policy", {}).get("class_predictions_used_as_targets") is not False
        or unlabeled.get("policy", {}).get("only_augmentation_consistency_allowed") is not True
    ):
        raise RuntimeError("Stage69 unlabeled consistency evidence is invalid")
    for name in ("stage67_builder_report", "stage67_near_report", "stage67_strata_report", "stage67_leakage_report"):
        report = json.loads(paths[name].read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            raise RuntimeError(f"Stage69 inherited audit did not pass: {name}")
    source_license = json.loads(paths["stage68_license_evidence"].read_text(encoding="utf-8"))
    if source_license.get("sources", [{}])[0].get("training_decision") != "countable_research_only_non_deployable":
        raise RuntimeError("Stage69 DVM research license isolation failed")
    finalization = json.loads(paths["stage68_finalization_report"].read_text(encoding="utf-8"))
    if finalization.get("status") != "pass" or finalization.get("output_train_rows") < 118338:
        raise RuntimeError("Stage69 DVM data finalization gate failed")
    return paths


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
        raise FileExistsError("refusing to overwrite Stage69 evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    paths = validate_matrix(matrix, args.matrix, Path(__file__).resolve())
    candidates = list(matrix.get("candidates", []))
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise RuntimeError("Stage69 candidate IDs are empty or duplicated")
    for candidate in candidates:
        for key in ("init_checkpoint", "teacher_checkpoint"):
            path = Path(candidate[key])
            if not path.is_file() or sha256(path).lower() != candidate[f"{key}_sha256"].lower():
                raise RuntimeError(f"Stage69 candidate dependency mismatch: {path}")
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only", "matrix_sha256": sha256(args.matrix),
            "candidate_count": len(candidates), "test_accessed": False,
            "frozen_video_used": False, "production_model_modified": False,
            "deployment_performed": False
        }, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "attribute-stage69-uvh-color-consistency-run-v1",
        "status": "running", "created_at": now(), "updated_at": now(),
        "matrix": str(args.matrix.resolve()), "matrix_sha256": sha256(args.matrix),
        "candidates_total": len(candidates), "completed": [], "failed": [],
        "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
        "deployment_paused_by_user": True,
        "training_eligibility": "research-only; non-deployable"
    }
    atomic_json(args.state, state)
    common = {
        **matrix["common"],
        "manifest": str(paths["supervised_manifest"].resolve()),
        "labels": str(paths["labels"].resolve()),
        "device": args.device,
        "dataset_version": "attribute-domain-v2-stage69-uvh-color-consistency-v1",
        "code_revision": args.code_revision,
    }
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        parameters = {**common, **candidate, "output_dir": str(output_dir)}
        command = [sys.executable, str(paths["training_script"]), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "started_at": now(), "command": command}
        state["active"] = active; state["updated_at"] = now(); atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(
                command, cwd=str(paths["training_script"].resolve().parents[1]),
                stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        best = output_dir / "best.pt"
        record = {
            **active, "finished_at": now(), "return_code": result.returncode,
            "stdout_log": str(log.resolve()), "stdout_log_sha256": sha256(log),
            "best_checkpoint": str(best.resolve()) if best.is_file() else None,
            "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
        }
        state["completed" if result.returncode == 0 and best.is_file() else "failed"].append(record)
        state.pop("active", None); state["updated_at"] = now(); atomic_json(args.state, state)
    state["status"] = "complete_validation_only"; state["updated_at"] = now(); atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}))
    return 0 if not state["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
