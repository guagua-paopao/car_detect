#!/usr/bin/env python3
"""Run production-eligible Stage66 architecture/adverse candidates."""

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


META_KEYS = {
    "candidate_id", "purpose", "init_checkpoint_sha256", "teacher_checkpoint_sha256"
}


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
        weighted.append((weight, night, small, occluded, truthy(row.get("stage66_licensed_adverse"))))
    total = sum(item[0] for item in weighted)
    if total <= 0:
        raise RuntimeError("Stage66 sampler has zero total weight")
    return {
        "night": sum(item[0] for item in weighted if item[1]) / total,
        "small": sum(item[0] for item in weighted if item[2]) / total,
        "occluded_or_truncated": sum(item[0] for item in weighted if item[3]) / total,
        "licensed_adverse": sum(item[0] for item in weighted if item[4]) / total,
    }


def validate_evidence_chain(
    manifest_hash: str,
    manifest_report: dict,
    dedup_report: dict,
    near_dedup_report: dict,
    strata_report: dict,
    leakage_report: dict,
) -> None:
    """Require every Stage66 audit to describe one immutable manifest lineage."""
    base_hash = str(manifest_report.get("output_manifest_sha256", "")).lower()
    if manifest_report.get("status") != "pass" or not base_hash:
        raise RuntimeError("Stage66 manifest evidence is invalid")
    if not manifest_report.get("evaluation_membership_preserved") or manifest_report.get("cross_split_group_overlap") != 0:
        raise RuntimeError("Stage66 split isolation failed")

    dedup_policy = dedup_report.get("policy", {})
    if (
        dedup_report.get("status") != "pass"
        or str(dedup_report.get("input_manifest_sha256", "")).lower() != base_hash
        or not str(dedup_report.get("output_manifest_sha256", "")).lower()
        or dedup_report.get("unreadable_images") != 0
        or not dedup_policy.get("exact_byte_duplicates_removed_from_train")
        or not dedup_policy.get("exact_cross_split_train_duplicates_removed")
        or not dedup_policy.get("heldout_membership_preserved")
    ):
        raise RuntimeError("Stage66 full-crop hash/dedup evidence is invalid")
    for key in ("test_labels_or_predictions_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if dedup_policy.get(key) is not False:
            raise RuntimeError(f"Stage66 hash/dedup policy violation: {key}")
    if dedup_policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage66 hash/dedup evidence lost the deployment pause")

    near_policy = near_dedup_report.get("policy", {})
    if (
        near_dedup_report.get("status") != "pass"
        or str(near_dedup_report.get("input_manifest_sha256", "")).lower()
        != str(dedup_report.get("output_manifest_sha256", "")).lower()
        or str(near_dedup_report.get("output_manifest_sha256", "")).lower() != manifest_hash
        or near_dedup_report.get("residual_cross_split_near_pairs") != 0
        or near_dedup_report.get("residual_exact_sha_cross_split_overlap") != 0
        or near_dedup_report.get("maximum_hamming_distance") != 4
        or not near_policy.get("all_train_rows_with_dhash_distance_le_4_to_heldout_removed")
        or not near_policy.get("heldout_membership_preserved")
        or not near_policy.get("heldout_labels_not_used_for_filtering")
    ):
        raise RuntimeError("Stage66 cross-split near-dedup evidence is invalid")
    for key in ("test_labels_or_predictions_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if near_policy.get(key) is not False:
            raise RuntimeError(f"Stage66 near-dedup policy violation: {key}")
    if near_policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage66 near-dedup evidence lost the deployment pause")

    strata_policy = strata_report.get("policy", {})
    if (
        strata_report.get("status") != "pass"
        or str(strata_report.get("manifest_sha256", "")).lower() != manifest_hash
        or strata_report.get("invalid_train_license_rows") != 0
        or strata_report.get("frozen_marker_rows") != 0
    ):
        raise RuntimeError("Stage66 final-manifest strata evidence is invalid")
    for key in ("test_labels_or_predictions_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if strata_policy.get(key) is not False:
            raise RuntimeError(f"Stage66 strata policy violation: {key}")
    if strata_policy.get("deployment_paused_by_user") is not True:
        raise RuntimeError("Stage66 strata evidence lost the deployment pause")

    leakage_policy = leakage_report.get("policy", {})
    if leakage_report.get("status") != "pass" or str(leakage_report.get("manifest_sha256", "")).lower() != manifest_hash:
        raise RuntimeError("Stage66 perceptual leakage evidence is invalid")
    if (
        leakage_report.get("exact_sha_cross_split_overlap") != 0
        or leakage_report.get("source_group_cross_split_overlap") != 0
        or leakage_report.get("dhash", {}).get("cross_split_near_pairs") != 0
    ):
        raise RuntimeError("Stage66 perceptual leakage gate failed")
    for key in ("test_labels_or_predictions_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if leakage_policy.get(key) is not False:
            raise RuntimeError(f"Stage66 leakage policy violation: {key}")


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
        raise FileExistsError("refusing to overwrite Stage66 evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()).upper() != matrix.get("runner_script_sha256", "").upper():
        raise RuntimeError("Stage66 runner script hash mismatch")
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage66 matrix is not prepared")
    policy = matrix.get("execution_policy", {})
    for key in ("iterative_test_access", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage66 execution policy violation: {key}")
    if policy.get("deployment_paused_by_user") is not True or policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage66 lost deployment pause or test isolation")
    if policy.get("license_policy") != "production_eligible_training_data_offline_deployment_paused":
        raise RuntimeError("Stage66 license policy is not production eligible")
    paths = {
        "training_script": Path(matrix["training_script"]),
        "labels": Path(matrix["labels"]),
        "manifest": Path(matrix["manifest"]),
        "manifest_report": Path(matrix["manifest_report"]),
        "dedup_report": Path(matrix["dedup_report"]),
        "near_dedup_report": Path(matrix["near_dedup_report"]),
        "strata_report": Path(matrix["strata_report"]),
        "leakage_report": Path(matrix["leakage_report"]),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("training_script", "labels", "manifest", "manifest_report", "dedup_report", "near_dedup_report", "strata_report", "leakage_report"):
        if sha256(paths[name]).upper() != matrix[f"{name}_sha256"].upper():
            raise RuntimeError(f"Stage66 {name} hash mismatch")
    for raw_path, expected in matrix.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path).upper() != expected.upper():
            raise RuntimeError(f"Stage66 source hash mismatch: {path}")
    manifest_hash = sha256(paths["manifest"])
    manifest_report = json.loads(paths["manifest_report"].read_text(encoding="utf-8"))
    dedup_report = json.loads(paths["dedup_report"].read_text(encoding="utf-8"))
    near_dedup_report = json.loads(paths["near_dedup_report"].read_text(encoding="utf-8"))
    strata_report = json.loads(paths["strata_report"].read_text(encoding="utf-8"))
    leakage = json.loads(paths["leakage_report"].read_text(encoding="utf-8"))
    validate_evidence_chain(manifest_hash, manifest_report, dedup_report, near_dedup_report, strata_report, leakage)
    with paths["manifest"].open("r", encoding="utf-8-sig", newline="") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row.get("split") == "train" and row.get("review_status") == "approved"]
    candidates = list(matrix.get("candidates", []))
    ids = [item.get("candidate_id") for item in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise RuntimeError("Stage66 candidate IDs are empty or duplicated")
    quota_reports = {}
    for candidate in candidates:
        parameters = {**matrix["common"], **candidate}
        if parameters.get("body_hierarchy") != "none":
            raise RuntimeError("Stage66 v1 candidates must preserve the flat production decode contract")
        if float(parameters.get("coarse_car_loss_weight", 0)) <= 0 or float(parameters.get("coarse_truck_loss_weight", 0)) <= 0:
            raise RuntimeError("Stage66 lost coarse family partial supervision")
        for field in ("init_checkpoint", "teacher_checkpoint"):
            if field not in candidate:
                continue
            path = Path(candidate[field])
            expected = candidate[f"{field}_sha256"]
            if not path.is_file() or sha256(path).upper() != expected.upper():
                raise RuntimeError(f"Stage66 candidate dependency mismatch: {path}")
        quotas = expected_sampler_quotas(train_rows, parameters)
        for metric, minimum in (
            ("night", matrix["quota_policy"]["minimum_expected_night_fraction"]),
            ("small", matrix["quota_policy"]["minimum_expected_small_fraction"]),
            ("occluded_or_truncated", matrix["quota_policy"]["minimum_expected_occluded_or_truncated_fraction"]),
        ):
            if quotas[metric] < minimum:
                raise RuntimeError(f"{candidate['candidate_id']} misses {metric} quota")
        quota_reports[candidate["candidate_id"]] = quotas
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only",
            "matrix_sha256": sha256(args.matrix),
            "manifest_sha256": sha256(paths["manifest"]),
            "dedup_report_sha256": sha256(paths["dedup_report"]),
            "near_dedup_report_sha256": sha256(paths["near_dedup_report"]),
            "strata_report_sha256": sha256(paths["strata_report"]),
            "leakage_report_sha256": sha256(paths["leakage_report"]),
            "expected_sampler_quotas": quota_reports,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        }, ensure_ascii=False, indent=2))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=False)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "attribute-stage66-licensed-adverse-run-v1", "status": "running",
        "created_at": now(), "updated_at": now(), "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix), "manifest_sha256": sha256(paths["manifest"]),
        "manifest_report_sha256": sha256(paths["manifest_report"]),
        "dedup_report_sha256": sha256(paths["dedup_report"]),
        "near_dedup_report_sha256": sha256(paths["near_dedup_report"]),
        "strata_report_sha256": sha256(paths["strata_report"]),
        "leakage_report_sha256": sha256(paths["leakage_report"]),
        "training_script_sha256": sha256(paths["training_script"]), "source_hashes": matrix["source_hashes"],
        "expected_sampler_quotas": quota_reports, "candidates_total": len(candidates), "completed": [], "failed": [],
        "test_accessed": False, "frozen_video_used": False, "production_model_modified": False,
        "deployment_performed": False, "deployment_paused_by_user": True,
    }
    atomic_json(args.state, state)
    common = {
        **matrix["common"], "manifest": str(paths["manifest"].resolve()), "labels": str(paths["labels"].resolve()),
        "device": args.device, "dataset_version": "attribute-domain-v2-stage66-licensed-adverse-v1", "code_revision": args.code_revision,
    }
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_dir = args.output_root / candidate_id
        command = [sys.executable, str(paths["training_script"]), *command_arguments({**common, **candidate, "output_dir": str(output_dir)})]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "started_at": now(), "command": command}
        state["active"] = active; state["updated_at"] = now(); atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(command, cwd=str(paths["training_script"].resolve().parents[1]), stdout=handle, stderr=subprocess.STDOUT, check=False)
        best = output_dir / "best.pt"
        record = {**active, "finished_at": now(), "return_code": result.returncode, "stdout_log": str(log.resolve()), "stdout_log_sha256": sha256(log), "best_checkpoint": str(best.resolve()) if best.is_file() else None, "best_checkpoint_sha256": sha256(best) if best.is_file() else None}
        state["completed" if result.returncode == 0 and best.is_file() else "failed"].append(record)
        state.pop("active", None); state["updated_at"] = now(); atomic_json(args.state, state)
    state["status"] = "complete_validation_only"; state["updated_at"] = now(); atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
