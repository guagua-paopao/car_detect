#!/usr/bin/env python3
"""Run the audited Stage62 validation-only candidate matrix on one GPU queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


META_KEYS = {"candidate_id", "purpose", "dependency"}


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


def validate_manifest_report(report_path: Path, manifest_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("final Stage62 manifest report did not pass")
    if report.get("output_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("final Stage62 manifest hash differs from report")
    policy = report.get("policy", {})
    required = {
        "validation_and_test_rows_unchanged": True,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"manifest report policy {key!r} is not {expected!r}")
    return report


def command_arguments(parameters: dict) -> list[str]:
    result: list[str] = []
    for key, value in parameters.items():
        if key in META_KEYS or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
            continue
        result.extend([flag, str(value)])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage62 candidate evidence")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_waiting_weather_combined_manifest":
        raise RuntimeError("candidate matrix is not in the expected prepared state")
    execution = matrix.get("execution_policy", {})
    if execution.get("iterative_test_access") is not False or execution.get("required_flag") != "--skip-test":
        raise RuntimeError("candidate matrix does not enforce validation-only screening")
    validate_manifest_report(args.manifest_report, args.manifest)
    training_script = Path(matrix["training_script"])
    labels = Path(matrix["labels"])
    if not training_script.is_file() or not labels.is_file():
        raise FileNotFoundError("training script or label ontology is missing")

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    common = dict(matrix["common"])
    common["manifest"] = str(args.manifest.resolve())
    common["labels"] = str(labels.resolve())
    common["device"] = args.device
    common["dataset_version"] = "attribute-domain-v2-stage62-weather-combined-v1"
    common["code_revision"] = args.code_revision
    candidates = [*matrix.get("screening_wave", []), *matrix.get("dependent_wave", [])]
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("candidate IDs are not unique")
    state = {
        "schema_version": "attribute-stage62-matrix-run-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "manifest_report": str(args.manifest_report.resolve()),
        "manifest_report_sha256": sha256(args.manifest_report),
        "training_script": str(training_script.resolve()),
        "training_script_sha256": sha256(training_script),
        "labels": str(labels.resolve()),
        "labels_sha256": sha256(labels),
        "device": args.device,
        "candidates_total": len(candidates),
        "completed": [],
        "failed_or_gate_rejected": [],
        "skipped_dependency": [],
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    atomic_json(args.state, state)

    output_by_id: dict[str, Path] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        if output_dir.exists():
            raise FileExistsError(f"candidate output already exists: {output_dir}")
        dependency = candidate.get("dependency")
        teacher_checkpoint = None
        if dependency:
            dependency_id = str(dependency).split("/", 1)[0]
            dependency_root = output_by_id.get(dependency_id)
            if dependency_root is None or not (dependency_root / "best.pt").is_file():
                state["skipped_dependency"].append({
                    "candidate_id": candidate_id,
                    "dependency": dependency,
                    "reason": "dependency best.pt is unavailable",
                })
                state["updated_at"] = now()
                atomic_json(args.state, state)
                continue
            teacher_checkpoint = dependency_root / "best.pt"
        parameters = {**common, **candidate, "output_dir": str(output_dir)}
        if teacher_checkpoint is not None:
            parameters["teacher_checkpoint"] = str(teacher_checkpoint)
        command = [sys.executable, str(training_script), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"candidate {candidate_id} lost mandatory --skip-test")
        command_record = {
            "candidate_id": candidate_id,
            "started_at": now(),
            "command": command,
            "output_dir": str(output_dir.resolve()),
        }
        state["active"] = command_record
        state["updated_at"] = now()
        atomic_json(args.state, state)
        log_path = args.output_root / f"{candidate_id}.stdout.log"
        with log_path.open("wb") as log_handle:
            result = subprocess.run(
                command,
                cwd=str(training_script.resolve().parents[2]),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        output_by_id[candidate_id] = output_dir
        best = output_dir / "best.pt"
        record = {
            **command_record,
            "finished_at": now(),
            "return_code": result.returncode,
            "stdout_log": str(log_path.resolve()),
            "stdout_log_sha256": sha256(log_path),
            "best_checkpoint": str(best.resolve()) if best.is_file() else None,
            "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
        }
        destination = "completed" if result.returncode == 0 and best.is_file() else "failed_or_gate_rejected"
        state[destination].append(record)
        state.pop("active", None)
        state["updated_at"] = now()
        atomic_json(args.state, state)

    state["status"] = "complete_screening_validation_only"
    state["updated_at"] = now()
    state["candidates_with_best_checkpoint"] = sum(
        1 for candidate_id in ids
        if candidate_id in output_by_id and (output_by_id[candidate_id] / "best.pt").is_file()
    )
    state["all_commands_enforced_skip_test"] = True
    atomic_json(args.state, state)
    print(json.dumps({
        "status": state["status"],
        "completed": len(state["completed"]),
        "failed_or_gate_rejected": len(state["failed_or_gate_rejected"]),
        "skipped_dependency": len(state["skipped_dependency"]),
        "candidates_with_best_checkpoint": state["candidates_with_best_checkpoint"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
