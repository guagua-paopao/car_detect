#!/usr/bin/env python3
"""Run the fail-closed Stage62 domain-consistency screening queue."""

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


def require_policy(report: dict, expected: dict[str, object], name: str) -> None:
    policy = report.get("policy", {})
    for key, value in expected.items():
        if policy.get(key) is not value:
            raise RuntimeError(f"{name} policy {key!r} is not {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--supervised-manifest", type=Path, required=True)
    parser.add_argument("--supervised-report", type=Path, required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--unlabeled-report", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--attribution-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage62 domain-consistency evidence")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("domain-consistency matrix is not prepared")
    execution = matrix.get("execution_policy", {})
    if execution.get("iterative_test_access") is not False or execution.get("required_flag") != "--skip-test":
        raise RuntimeError("matrix does not enforce validation-only screening")
    if execution.get("deployment_paused_by_user") is not True:
        raise RuntimeError("matrix lost the user deployment pause")

    supervised_report = json.loads(args.supervised_report.read_text(encoding="utf-8"))
    if supervised_report.get("status") != "pass_preweather_only":
        raise RuntimeError("supervised Stage59 report is not the locked pass artifact")
    if supervised_report.get("output_manifest_sha256") != sha256(args.supervised_manifest):
        raise RuntimeError("supervised manifest hash mismatch")
    require_policy(
        supervised_report,
        {
            "validation_and_test_rows_unchanged": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "supervised",
    )

    unlabeled_report = json.loads(args.unlabeled_report.read_text(encoding="utf-8"))
    if unlabeled_report.get("status") != "pass":
        raise RuntimeError("unlabeled adverse-scene report did not pass")
    if unlabeled_report.get("output_manifest_sha256") != sha256(args.unlabeled_manifest):
        raise RuntimeError("unlabeled manifest hash mismatch")
    require_policy(
        unlabeled_report,
        {
            "all_rows_train_only": True,
            "all_attributes_remain_unknown_unsupervised": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "unlabeled",
    )
    rows = int(unlabeled_report.get("rows", 0))
    if rows < 3000 or float(unlabeled_report.get("night_fraction", 0.0)) < 0.30:
        raise RuntimeError("unlabeled stream misses the size or real-night quota")
    if int(unlabeled_report.get("small_rows", 0)) / rows < 0.20:
        raise RuntimeError("unlabeled stream misses the small-target quota")
    hard_rows = max(
        int(unlabeled_report.get("occluded_rows", 0)),
        int(unlabeled_report.get("truncated_rows", 0)),
    )
    if hard_rows / rows < 0.15:
        raise RuntimeError("unlabeled stream misses the hard-condition quota")

    attribution_report = json.loads(args.attribution_report.read_text(encoding="utf-8"))
    if attribution_report.get("status") != "pass":
        raise RuntimeError("per-image attribution report did not pass")
    if attribution_report.get("input_manifest_sha256") != sha256(args.unlabeled_manifest):
        raise RuntimeError("attribution input hash mismatch")
    if attribution_report.get("output_attribution_sha256") != sha256(args.attribution):
        raise RuntimeError("attribution CSV hash mismatch")
    require_policy(
        attribution_report,
        {
            "per_image_cc_by_2_url_required": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "attribution",
    )

    training_script = Path(matrix["training_script"])
    labels = Path(matrix["labels"])
    if not training_script.is_file() or not labels.is_file():
        raise FileNotFoundError("training script or labels are missing")
    candidates = list(matrix.get("candidates", []))
    ids = [candidate["candidate_id"] for candidate in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise RuntimeError("candidate IDs are empty or not unique")

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "attribute-stage62-domain-consistency-run-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "supervised_manifest": str(args.supervised_manifest.resolve()),
        "supervised_manifest_sha256": sha256(args.supervised_manifest),
        "supervised_report_sha256": sha256(args.supervised_report),
        "unlabeled_manifest": str(args.unlabeled_manifest.resolve()),
        "unlabeled_manifest_sha256": sha256(args.unlabeled_manifest),
        "unlabeled_report_sha256": sha256(args.unlabeled_report),
        "attribution_sha256": sha256(args.attribution),
        "attribution_report_sha256": sha256(args.attribution_report),
        "training_script_sha256": sha256(training_script),
        "candidates_total": len(candidates),
        "completed": [],
        "failed": [],
        "skipped_dependency": [],
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    common = {
        **matrix["common"],
        "manifest": str(args.supervised_manifest.resolve()),
        "labels": str(labels.resolve()),
        "unlabeled_manifest": str(args.unlabeled_manifest.resolve()),
        "unlabeled_root": str(Path(matrix["unlabeled_root"]).resolve()),
        "device": args.device,
        "dataset_version": "attribute-domain-v2-stage62-domain-consistency-v1",
        "code_revision": args.code_revision,
    }

    output_by_id: dict[str, Path] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
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
        output_by_id[candidate_id] = output_dir
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
