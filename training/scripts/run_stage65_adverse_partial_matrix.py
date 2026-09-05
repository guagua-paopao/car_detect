#!/usr/bin/env python3
"""Run isolated Stage65 partial-label candidates after Stage64 validation."""
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


META_KEYS = {"candidate_id", "purpose"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_arguments(parameters: dict) -> list[str]:
    result = []
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


def expected_sampler_quotas(rows: list[dict[str, str]], parameters: dict) -> dict[str, float]:
    weighted = []
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
        if small:
            weight *= 1.0 + float(parameters.get("small_sample_weight", 0.0))
        color_supervised = str(row.get("color_supervised", "true")).lower() not in {"false", "0", "no"}
        color_supervised = color_supervised and row.get("color") not in {"", "unknown", None}
        if color_supervised:
            weight *= 1.0 + float(parameters.get("color_sample_weight", 0.0))
        night = truthy(row.get("night"))
        occluded = truthy(row.get("occluded")) or truthy(row.get("truncated"))
        if night:
            weight *= float(parameters.get("night_sample_weight", 1.0))
        if occluded:
            weight *= float(parameters.get("occlusion_sample_weight", 1.0))
        weighted.append((weight, night, small, occluded, truthy(row.get("adverse_supervised"))))
    total = sum(item[0] for item in weighted)
    if total <= 0:
        raise RuntimeError("Stage65 sampler has zero total weight")
    return {
        "night": sum(item[0] for item in weighted if item[1]) / total,
        "small": sum(item[0] for item in weighted if item[2]) / total,
        "occluded_or_truncated": sum(item[0] for item in weighted if item[3]) / total,
        "adverse_supervised": sum(item[0] for item in weighted if item[4]) / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage65 evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage65 matrix is not prepared")
    policy = matrix.get("execution_policy", {})
    for key in ("iterative_test_access", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage65 execution policy violation: {key}")
    if policy.get("deployment_paused_by_user") is not True or policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage65 lost deployment pause or test isolation")
    if policy.get("license_policy") != "offline_research_only_no_deployment":
        raise RuntimeError("Stage65 license policy is not fail closed")
    if policy.get("exact_body_gate_not_replaced_by_family_metric") is not True:
        raise RuntimeError("Stage65 may substitute a family metric for the exact gate")

    training_script = Path(matrix["training_script"])
    labels = Path(matrix["labels"])
    manifest = Path(matrix["manifest"])
    report_path = Path(matrix["manifest_report"])
    leakage_report_path = Path(matrix["leakage_report"])
    for path in (training_script, labels, manifest, report_path, leakage_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(training_script).upper() != matrix["training_script_sha256"].upper():
        raise RuntimeError("Stage65 training script hash mismatch")
    if sha256(labels).upper() != matrix["labels_sha256"].upper():
        raise RuntimeError("Stage65 labels hash mismatch")
    for raw_path, expected in matrix.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path).upper() != expected.upper():
            raise RuntimeError(f"Stage65 source hash mismatch: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("output_manifest_sha256", "").lower() != sha256(manifest):
        raise RuntimeError("Stage65 manifest evidence is invalid")
    manifest_policy = report.get("policy", {})
    for key in ("official_car_is_partial_family_not_exact_subtype", "official_truck_is_hierarchical_coarse_truth", "unsupported_merged_color_truth_rejected", "validation_and_test_membership_preserved"):
        if manifest_policy.get(key) is not True:
            raise RuntimeError(f"Stage65 manifest policy violation: {key}")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if manifest_policy.get(key) is not False:
            raise RuntimeError(f"Stage65 manifest policy violation: {key}")
    leakage = json.loads(leakage_report_path.read_text(encoding="utf-8"))
    if leakage.get("status") != "pass" or leakage.get("manifest_sha256", "").lower() != sha256(manifest):
        raise RuntimeError("Stage65 perceptual leakage audit did not pass")
    if leakage.get("exact_sha_cross_split_overlap") != 0 or leakage.get("source_group_cross_split_overlap") != 0:
        raise RuntimeError("Stage65 leakage audit found exact or group overlap")
    if leakage.get("dhash", {}).get("cross_split_near_pairs") != 0:
        raise RuntimeError("Stage65 leakage audit found perceptual cross-split overlap")
    leakage_policy = leakage.get("policy", {})
    for key in ("test_labels_or_predictions_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if leakage_policy.get(key) is not False:
            raise RuntimeError(f"Stage65 leakage-audit policy violation: {key}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row.get("split") == "train" and row.get("review_status") == "approved"]
    candidates = list(matrix.get("candidates", []))
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise RuntimeError("Stage65 candidate IDs are empty or duplicated")
    quota_policy = matrix["quota_policy"]
    quota_reports = {}
    for candidate in candidates:
        parameters = {**matrix["common"], **candidate}
        if parameters.get("body_hierarchy") != "truck_family" or float(parameters.get("coarse_car_loss_weight", 0)) <= 0:
            raise RuntimeError(f"{candidate['candidate_id']} lost partial-label hierarchy")
        init = Path(candidate["init_checkpoint"])
        if not init.is_file():
            raise FileNotFoundError(init)
        quotas = expected_sampler_quotas(train_rows, parameters)
        if quotas["night"] < quota_policy["minimum_expected_night_fraction"]:
            raise RuntimeError(f"{candidate['candidate_id']} misses night quota")
        if quotas["small"] < quota_policy["minimum_expected_small_fraction"]:
            raise RuntimeError(f"{candidate['candidate_id']} misses small-target quota")
        if quotas["occluded_or_truncated"] < quota_policy["minimum_expected_occluded_or_truncated_fraction"]:
            raise RuntimeError(f"{candidate['candidate_id']} misses occlusion quota")
        quota_reports[candidate["candidate_id"]] = quotas

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "stage65-adverse-partial-run-v1", "status": "running",
        "created_at": now(), "updated_at": now(), "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix), "manifest_sha256": sha256(manifest),
        "manifest_report_sha256": sha256(report_path), "training_script_sha256": sha256(training_script),
        "leakage_report_sha256": sha256(leakage_report_path),
        "source_hashes": matrix["source_hashes"], "expected_sampler_quotas": quota_reports,
        "candidates_total": len(candidates), "completed": [], "failed": [],
        "test_accessed": False, "frozen_video_used": False, "production_model_modified": False,
        "deployment_performed": False, "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    common = {**matrix["common"], "manifest": str(manifest.resolve()), "labels": str(labels.resolve()), "device": args.device, "dataset_version": "attribute-domain-v2-stage65-adverse-partial-v1", "code_revision": args.code_revision}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        command = [sys.executable, str(training_script), *command_arguments({**common, **candidate, "output_dir": str(output_dir)})]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "started_at": now(), "command": command}
        state["active"] = active; state["updated_at"] = now(); atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(command, cwd=str(training_script.resolve().parents[2]), stdout=handle, stderr=subprocess.STDOUT, check=False)
        best = output_dir / "best.pt"
        record = {**active, "finished_at": now(), "return_code": result.returncode, "stdout_log": str(log.resolve()), "stdout_log_sha256": sha256(log), "best_checkpoint": str(best.resolve()) if best.is_file() else None, "best_checkpoint_sha256": sha256(best) if best.is_file() else None}
        state["completed" if result.returncode == 0 and best.is_file() else "failed"].append(record)
        state.pop("active", None); state["updated_at"] = now(); atomic_json(args.state, state)
    state["status"] = "complete_validation_only"; state["updated_at"] = now(); atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
