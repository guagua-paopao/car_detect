#!/usr/bin/env python3
"""Run the conditional, production-initialized Stage67 UVH-26 type follow-up."""

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


META_KEYS = {"candidate_id", "purpose", "init_checkpoint_sha256", "teacher_checkpoint_sha256"}
SAFE_FALSE_KEYS = ("frozen_video_used", "production_model_modified", "deployment_performed")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def source_group_key(row: dict[str, str]) -> str:
    return next(
        (str(row.get(key, "")).strip() for key in ("source_group", "track_group", "source_frame_id", "source_image_id") if str(row.get(key, "")).strip()),
        "",
    )


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


def require_safe_policy(policy: dict, context: str) -> None:
    for key in SAFE_FALSE_KEYS:
        if policy.get(key) is not False:
            raise RuntimeError(f"{context} policy violation: {key}")
    test_keys = [key for key in ("test_accessed", "test_labels_or_predictions_accessed") if key in policy]
    if not test_keys or any(policy.get(key) is not False for key in test_keys):
        raise RuntimeError(f"{context} policy violation: test access")
    if policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError(f"{context} lost deployment pause")


def validate_decision(decision: dict, report_path: Path) -> None:
    if decision.get("status") != "pass" or decision.get("action") != "run_stage67_uvh26_controlled_type_followup":
        raise RuntimeError("Stage67 was not authorized by the conditional type-gap decision")
    require_safe_policy(decision.get("policy", {}), "Stage67 decision")
    if str(decision.get("stage66_validation_report_sha256", "")).lower() != sha256(report_path):
        raise RuntimeError("Stage67 decision does not pin the supplied Stage66 validation report")
    source = decision.get("recommended_stage67_initialization_source", {})
    if source.get("type_gate_pass") is not False:
        raise RuntimeError("Stage67 decision does not demonstrate a remaining type gap")


def validate_evidence(matrix: dict, paths: dict[str, Path]) -> dict[str, int]:
    manifest_hash = sha256(paths["manifest"])
    builder = json.loads(paths["builder_report"].read_text(encoding="utf-8"))
    near = json.loads(paths["near_dedup_report"].read_text(encoding="utf-8"))
    strata = json.loads(paths["strata_report"].read_text(encoding="utf-8"))
    leakage = json.loads(paths["leakage_report"].read_text(encoding="utf-8"))
    license_evidence = json.loads(paths["license_evidence"].read_text(encoding="utf-8"))

    if builder.get("status") != "pass":
        raise RuntimeError("Stage67 controlled-manifest builder did not pass")
    require_safe_policy(builder.get("policy", {}), "Stage67 builder")
    if str(builder.get("base_manifest_sha256", "")).lower() != matrix["base_manifest_sha256"].lower():
        raise RuntimeError("Stage67 builder does not descend from the pinned Stage66 manifest")
    if str(builder.get("license_evidence_sha256", "")).lower() != sha256(paths["license_evidence"]):
        raise RuntimeError("Stage67 builder license evidence mismatch")
    if int(builder.get("selected_exact_color_rows", -1)) != 0:
        raise RuntimeError("Stage67 UVH selection unexpectedly claims exact color truth")

    if near.get("status") != "pass" or str(near.get("input_manifest_sha256", "")).lower() != str(builder.get("output_manifest_sha256", "")).lower():
        raise RuntimeError("Stage67 near-dedup input lineage is invalid")
    if str(near.get("output_manifest_sha256", "")).lower() != manifest_hash:
        raise RuntimeError("Stage67 near-dedup output does not match the final manifest")
    if near.get("residual_cross_split_near_pairs") != 0 or near.get("residual_exact_sha_cross_split_overlap") != 0:
        raise RuntimeError("Stage67 near-duplicate leakage remains")
    require_safe_policy(near.get("policy", {}), "Stage67 near-dedup")

    if strata.get("status") != "pass" or str(strata.get("manifest_sha256", "")).lower() != manifest_hash:
        raise RuntimeError("Stage67 strata evidence is invalid")
    if strata.get("invalid_train_license_rows") != 0 or strata.get("frozen_marker_rows") != 0:
        raise RuntimeError("Stage67 final manifest contains invalid license or frozen markers")
    require_safe_policy(strata.get("policy", {}), "Stage67 strata")

    if leakage.get("status") != "pass" or str(leakage.get("manifest_sha256", "")).lower() != manifest_hash:
        raise RuntimeError("Stage67 leakage evidence is invalid")
    if leakage.get("exact_sha_cross_split_overlap") != 0 or leakage.get("source_group_cross_split_overlap") != 0:
        raise RuntimeError("Stage67 exact or source-group leakage remains")
    if leakage.get("dhash", {}).get("cross_split_near_pairs") != 0:
        raise RuntimeError("Stage67 perceptual leakage remains")
    require_safe_policy(leakage.get("policy", {}), "Stage67 leakage")

    if license_evidence.get("status") != "pass_primary_source_verified":
        raise RuntimeError("Stage67 UVH-26 primary-source license is not verified")
    require_safe_policy(license_evidence.get("policy", {}), "Stage67 license")
    if license_evidence.get("pinned_revision") != matrix["uvh26_pinned_revision"]:
        raise RuntimeError("Stage67 UVH-26 revision mismatch")
    if license_evidence.get("decision", {}).get("production_training_license_eligible") is not True:
        raise RuntimeError("Stage67 UVH-26 is not production-training eligible")

    with paths["manifest"].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "train" and row.get("review_status") == "approved"]
    uvh = [row for row in rows if row.get("source_dataset") == "UVH-26"]
    if len(uvh) != matrix["expected_uvh26_train_rows"]:
        raise RuntimeError("Stage67 UVH-26 row count mismatch")
    if any(row.get("body_type") in {"", "unknown"} or not truthy(row.get("body_type_supervised")) for row in uvh):
        raise RuntimeError("Stage67 UVH-26 rows lost exact body supervision")
    if any(row.get("color") not in {"", "unknown"} or truthy(row.get("color_supervised")) for row in uvh):
        raise RuntimeError("Stage67 UVH-26 rows unexpectedly contain exact color supervision")
    unique_groups = len({source_group_key(row) for row in uvh if source_group_key(row)})
    if unique_groups < matrix["minimum_uvh26_source_groups"]:
        raise RuntimeError("Stage67 UVH-26 source-group diversity is below the floor")
    return {"train_rows": len(rows), "uvh26_rows": len(uvh), "uvh26_source_groups": unique_groups}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--stage66-validation-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage67 evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()).upper() != matrix.get("runner_script_sha256", "").upper():
        raise RuntimeError("Stage67 runner script hash mismatch")
    if matrix.get("status") != "prepared_conditional_validation_only":
        raise RuntimeError("Stage67 matrix is not conditionally prepared")
    policy = matrix.get("execution_policy", {})
    require_safe_policy(policy, "Stage67 execution")
    if policy.get("iterative_test_access") is not False or policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage67 lost validation/test isolation")
    if policy.get("requires_stage66_type_gap_decision") is not True:
        raise RuntimeError("Stage67 lost its conditional trigger")

    paths = {name: Path(matrix[name]) for name in (
        "training_script", "labels", "manifest", "builder_report", "near_dedup_report",
        "strata_report", "leakage_report", "license_evidence",
    )}
    for name, path in paths.items():
        if not path.is_file() or sha256(path).upper() != matrix[f"{name}_sha256"].upper():
            raise RuntimeError(f"Stage67 {name} missing or hash mismatch: {path}")
    for raw_path, expected in matrix.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path).upper() != expected.upper():
            raise RuntimeError(f"Stage67 source hash mismatch: {path}")
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    validate_decision(decision, args.stage66_validation_report.resolve())
    evidence_counts = validate_evidence(matrix, paths)

    candidates = list(matrix.get("candidates", []))
    ids = [item.get("candidate_id") for item in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise RuntimeError("Stage67 candidate IDs are empty or duplicated")
    for candidate in candidates:
        if candidate.get("architecture") != "mobilenet_v3_large":
            raise RuntimeError("Stage67 conditional v1 must preserve the production architecture")
        if candidate.get("init_checkpoint") != candidate.get("teacher_checkpoint"):
            raise RuntimeError("Stage67 candidates must use the same production init and teacher")
        for field in ("init_checkpoint", "teacher_checkpoint"):
            path = Path(candidate[field])
            if not path.is_file() or sha256(path).upper() != candidate[f"{field}_sha256"].upper():
                raise RuntimeError(f"Stage67 candidate dependency mismatch: {path}")
        if float(candidate.get("distill_weight", 0.0)) <= 0:
            raise RuntimeError("Stage67 candidates must retain production-teacher distillation")

    preflight = {
        "status": "pass_preflight_only", "matrix_sha256": sha256(args.matrix),
        "decision_sha256": sha256(args.decision), "manifest_sha256": sha256(paths["manifest"]),
        "evidence_counts": evidence_counts, "candidate_count": len(candidates),
        "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "attribute-stage67-uvh26-controlled-run-v1", "status": "running",
        "created_at": now(), "updated_at": now(), "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix), "decision": str(args.decision.resolve()),
        "decision_sha256": sha256(args.decision), "stage66_validation_report_sha256": sha256(args.stage66_validation_report),
        "manifest_sha256": sha256(paths["manifest"]), "training_script_sha256": sha256(paths["training_script"]),
        "evidence_counts": evidence_counts, "candidates_total": len(candidates), "completed": [], "failed": [],
        "test_accessed": False, "frozen_video_used": False, "production_model_modified": False,
        "deployment_performed": False, "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    common = {
        **matrix["common"], "manifest": str(paths["manifest"].resolve()), "labels": str(paths["labels"].resolve()),
        "device": args.device, "dataset_version": "attribute-domain-v2-stage67-uvh26-controlled-v1",
        "code_revision": args.code_revision,
    }
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        command = [sys.executable, str(paths["training_script"]), *command_arguments({**common, **candidate, "output_dir": str(output_dir)})]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "started_at": now(), "command": command}
        state["active"] = active
        state["updated_at"] = now()
        atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(command, cwd=str(paths["training_script"].resolve().parents[1]), stdout=handle, stderr=subprocess.STDOUT, check=False)
        best = output_dir / "best.pt"
        record = {
            **active, "finished_at": now(), "return_code": result.returncode,
            "stdout_log": str(log.resolve()), "stdout_log_sha256": sha256(log),
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
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
