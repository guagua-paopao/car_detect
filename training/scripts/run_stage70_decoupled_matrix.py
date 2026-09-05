#!/usr/bin/env python3
"""Run fail-closed body/color specialist matrices with pinned evidence."""

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
    "candidate_id", "purpose", "specialist", "init_checkpoint_sha256",
    "teacher_checkpoint_sha256", "body_teacher_checkpoint_sha256",
    "color_teacher_checkpoint_sha256",
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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usable_train_rows(rows: list[dict[str, str]], specialist: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split") == "train"]
    if any("review_status" in row for row in selected):
        selected = [row for row in selected if row.get("review_status") == "approved"]
    if specialist == "body":
        return [row for row in selected if (
            (truthy(row.get("body_type_supervised")) and row.get("body_type") != "unknown")
            or row.get("coarse_body_family") in {"car", "truck"}
        )]
    if specialist == "color":
        return [row for row in selected if (
            truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown", None}
        )]
    raise ValueError(f"invalid specialist: {specialist}")


def expected_supervised_quota(rows: list[dict[str, str]], parameters: dict) -> dict[str, float]:
    weighted: list[tuple[float, bool, bool, bool]] = []
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
        if truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown", None}:
            weight *= 1.0 + float(parameters.get("color_sample_weight", 0.0))
        night = truthy(row.get("night"))
        occluded = truthy(row.get("occluded")) or truthy(row.get("truncated"))
        if night:
            weight *= float(parameters.get("night_sample_weight", 1.0))
        if occluded:
            weight *= float(parameters.get("occlusion_sample_weight", 1.0))
        weighted.append((weight, night, small, occluded))
    total = sum(item[0] for item in weighted)
    if total <= 0:
        raise RuntimeError("supervised sampler has zero total weight")
    return {
        "rows": len(rows),
        "night": sum(item[0] for item in weighted if item[1]) / total,
        "small": sum(item[0] for item in weighted if item[2]) / total,
        "occluded_or_truncated": sum(item[0] for item in weighted if item[3]) / total,
    }


def expected_unlabeled_night(rows: list[dict[str, str]], weight: float) -> float:
    train = [row for row in rows if row.get("split") == "train"]
    if not train or any(
        row.get("body_type") != "unknown" or row.get("color") != "unknown"
        or truthy(row.get("body_type_supervised")) or truthy(row.get("color_supervised"))
        for row in train
    ):
        raise RuntimeError("unlabeled stream is empty or contains a class target")
    night = sum(truthy(row.get("night")) for row in train)
    return (night * weight) / (night * weight + len(train) - night)


def validate_stage71_teacher_evidence(inputs: dict[str, Path], matrix: dict) -> dict:
    validation = json.loads(inputs["teacher_validation_report"].read_text(encoding="utf-8"))
    if validation.get("status") != "pass_pairs_available" or not validation.get("passing_pairs"):
        raise RuntimeError("Stage71 teacher validation produced no passing pair")
    policy = validation.get("policy", {})
    required_policy = {
        "validation_only": True,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"Stage71 teacher validation policy mismatch: {key}")
    pair_id = matrix.get("teacher_pair_id")
    pair = next((item for item in validation.get("pairs", []) if item.get("pair_id") == pair_id), None)
    if pair is None or pair.get("screen_status") != "pass" or pair_id not in validation["passing_pairs"]:
        raise RuntimeError("Stage71 selected teacher pair did not pass every validation gate")

    training = json.loads(inputs["teacher_training_state"].read_text(encoding="utf-8"))
    if training.get("status") != "complete_validation_only" or training.get("failed") or len(training.get("completed", [])) != 2:
        raise RuntimeError("Stage71 teacher training evidence is incomplete")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if training.get(key) is not False:
            raise RuntimeError(f"Stage71 teacher training policy mismatch: {key}")

    manifests = json.loads(inputs["teacher_manifest_report"].read_text(encoding="utf-8"))
    if manifests.get("status") != "pass" or manifests.get("eligibility") != "research-only_non-deployable":
        raise RuntimeError("Stage71 teacher manifest evidence did not pass")
    if manifests["outputs"]["color_teacher"]["splits"].get("train", 0) < 118338:
        raise RuntimeError("Stage71 color replay evidence fell below 118338")
    if manifests.get("post_filter_cross_split_near_leaks") or manifests.get("post_filter_group_leaks"):
        raise RuntimeError("Stage71 teacher manifests retain a split leak")
    return pair


def validate_teacher_binding(candidate: dict, teacher_pair: dict | None) -> None:
    if teacher_pair is None:
        return
    specialist = candidate.get("specialist")
    if specialist == "body":
        expected_path = teacher_pair["body_checkpoint"]
        expected_sha = teacher_pair["body_checkpoint_sha256"]
        actual_path = candidate.get("body_teacher_checkpoint")
        actual_sha = candidate.get("body_teacher_checkpoint_sha256")
        if candidate.get("color_teacher_checkpoint") is not None:
            raise RuntimeError("body specialist unexpectedly received a color teacher")
    elif specialist == "color":
        expected_path = teacher_pair["color_checkpoint"]
        expected_sha = teacher_pair["color_checkpoint_sha256"]
        actual_path = candidate.get("color_teacher_checkpoint")
        actual_sha = candidate.get("color_teacher_checkpoint_sha256")
        if candidate.get("body_teacher_checkpoint") is not None:
            raise RuntimeError("color specialist unexpectedly received a body teacher")
    else:
        raise RuntimeError("Stage71 student must be a body or color specialist")
    if actual_path != expected_path or str(actual_sha or "").lower() != str(expected_sha).lower():
        raise RuntimeError("student teacher binding does not match the passing validation pair")


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
        raise FileExistsError("refusing to overwrite Stage70 evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("specialist matrix is not prepared")
    if matrix.get("runner_sha256", "").lower() != sha256(Path(__file__).resolve()):
        raise RuntimeError("specialist runner SHA256 mismatch")
    evidence_mode = matrix.get("evidence_mode", "stage70_no_color_pseudo")
    if evidence_mode not in {"stage70_no_color_pseudo", "stage71_teacher_distillation"}:
        raise RuntimeError(f"unsupported specialist evidence mode: {evidence_mode}")
    policy = matrix.get("execution_policy", {})
    required = {
        "iterative_test_access": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
        "deployment_paused_by_user": True, "ua_color_pseudo_used": False,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"specialist execution policy mismatch: {key}")
    if policy.get("required_flag") != "--skip-test":
        raise RuntimeError("specialist matrix lost test isolation")
    if evidence_mode == "stage71_teacher_distillation" and policy.get("research_only") is not True:
        raise RuntimeError("Stage71 student matrix lost research-only isolation")

    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"specialist immutable input mismatch: {name}")
        inputs[name] = path
    needed = {"training_script", "labels", "body_manifest", "color_manifest", "unlabeled_manifest"}
    if evidence_mode == "stage70_no_color_pseudo":
        needed |= {"specialist_report", "failed_color_audit", "dvm_pool_report"}
    else:
        needed |= {"teacher_validation_report", "teacher_training_state", "teacher_manifest_report"}
    if not needed.issubset(inputs):
        raise RuntimeError("specialist immutable input set is incomplete")
    teacher_pair = None
    if evidence_mode == "stage70_no_color_pseudo":
        report = json.loads(inputs["specialist_report"].read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            raise RuntimeError("Stage70 specialist manifest report did not pass")
        report_policy = report.get("policy", {})
        if (
            report_policy.get("failed_color_pseudo_never_imported") is not True
            or report_policy.get("test_rows_imported") is not False
            or report_policy.get("frozen_video_used") is not False
        ):
            raise RuntimeError("Stage70 specialist manifest policy is invalid")
        for name in ("body", "color", "unlabeled"):
            if report["outputs"][name]["sha256"].lower() != sha256(inputs[f"{name}_manifest"]):
                raise RuntimeError(f"Stage70 report does not bind the {name} manifest")
        failed_audit = json.loads(inputs["failed_color_audit"].read_text(encoding="utf-8"))
        if failed_audit.get("status") != "fail":
            raise RuntimeError("Stage70 no-pseudo training lacks its preserved failed audit")
        dvm = json.loads(inputs["dvm_pool_report"].read_text(encoding="utf-8"))
        if dvm.get("status") != "pass" or dvm.get("split_counts", {}).get("train", 0) < 118338:
            raise RuntimeError("Stage70 DVM color pretraining evidence is insufficient")
    else:
        teacher_pair = validate_stage71_teacher_evidence(inputs, matrix)
        manifests = json.loads(inputs["teacher_manifest_report"].read_text(encoding="utf-8"))
        if manifests["outputs"]["body_teacher"]["sha256"].lower() != sha256(inputs["body_manifest"]):
            raise RuntimeError("Stage71 report does not bind the body manifest")
        if manifests["outputs"]["color_teacher"]["sha256"].lower() != sha256(inputs["color_manifest"]):
            raise RuntimeError("Stage71 report does not bind the color manifest")

    safety_root = Path(matrix.get("datasets_safety_root", "")).resolve()
    if not safety_root.is_dir():
        raise RuntimeError("Stage70 datasets safety root is missing")

    manifests = {name: inputs[f"{name}_manifest"] for name in ("body", "color", "unlabeled")}
    rows = {name: load_rows(path) for name, path in manifests.items()}
    for name, manifest_rows in rows.items():
        for row in manifest_rows:
            if row.get("split") == "test":
                raise RuntimeError(f"{name} manifest contains a test row")
            lowered = " ".join(str(value).lower() for value in row.values())
            if any(marker in lowered for marker in ("vcas_rtsp_demo_60s", "frozen_video", "36-48")):
                raise RuntimeError(f"{name} manifest contains a frozen-asset marker")
            raw = Path(row.get("image_path", ""))
            resolved = raw.resolve() if raw.is_absolute() else (manifests[name].parent / raw).resolve()
            try:
                resolved.relative_to(safety_root)
            except ValueError as error:
                raise RuntimeError(f"{name} image escapes the datasets safety root: {resolved}") from error
    candidates = list(matrix.get("candidates", []))
    if not candidates or len({item.get("candidate_id") for item in candidates}) != len(candidates):
        raise RuntimeError("Stage70 candidate IDs are empty or duplicated")
    quota_reports = {}
    for candidate in candidates:
        validate_teacher_binding(candidate, teacher_pair)
        specialist = candidate.get("specialist")
        parameters = {**matrix["common"], **candidate}
        if specialist == "body":
            valid = (
                parameters.get("selection_head") == "body"
                and float(parameters.get("body_loss_weight", 0)) > 0
                and float(parameters.get("color_loss_weight", -1)) == 0
                and float(parameters.get("unlabeled_body_weight", 0)) > 0
                and float(parameters.get("unlabeled_color_weight", -1)) == 0
            )
        elif specialist == "color":
            valid = (
                parameters.get("selection_head") == "color"
                and float(parameters.get("color_loss_weight", 0)) > 0
                and float(parameters.get("body_loss_weight", -1)) == 0
                and float(parameters.get("unlabeled_color_weight", 0)) > 0
                and float(parameters.get("unlabeled_body_weight", -1)) == 0
            )
        else:
            valid = False
        if not valid:
            raise RuntimeError(f"{candidate.get('candidate_id')} lost specialist head isolation")
        for key in (
            "init_checkpoint", "teacher_checkpoint",
            "body_teacher_checkpoint", "color_teacher_checkpoint",
        ):
            if not candidate.get(key):
                continue
            path = Path(candidate[key])
            if not path.is_file() or sha256(path).lower() != candidate[f"{key}_sha256"].lower():
                raise RuntimeError(f"specialist checkpoint mismatch: {path}")
        supervised = expected_supervised_quota(usable_train_rows(rows[specialist], specialist), parameters)
        unlabeled_night = expected_unlabeled_night(
            rows["unlabeled"], float(parameters.get("unlabeled_night_sample_weight", 1.0))
        )
        minimum = float(matrix["quota_policy"]["minimum_expected_night_fraction"])
        if supervised["night"] < minimum or unlabeled_night < minimum:
            raise RuntimeError(f"{candidate['candidate_id']} misses the sampled night quota")
        quota_reports[candidate["candidate_id"]] = {
            "supervised": supervised, "unlabeled_night": unlabeled_night,
        }
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only", "candidates": len(candidates),
            "expected_sampler_quotas": quota_reports, "test_accessed": False,
            "frozen_video_used": False, "production_model_modified": False,
            "deployment_performed": False,
        }, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": "specialist-matrix-run-v2", "status": "running",
        "created_at": now(), "updated_at": now(), "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix), "expected_sampler_quotas": quota_reports,
        "evidence_mode": evidence_mode,
        "candidates_total": len(candidates), "completed": [], "failed": [],
        "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "deployment_performed": False,
        "deployment_paused_by_user": True, "eligibility": "research-only_non-deployable",
    }
    args.state.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.state, state)
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        specialist = candidate["specialist"]
        output_dir = args.output_root / candidate_id
        parameters = {
            **matrix["common"], **candidate,
            "manifest": str(manifests[specialist].resolve()),
            "unlabeled_manifest": str(manifests["unlabeled"].resolve()),
            "unlabeled_root": str(safety_root),
            "labels": str(inputs["labels"].resolve()), "device": args.device,
            "dataset_version": matrix.get("dataset_version", "attribute-domain-v2-specialists-v1"),
            "code_revision": args.code_revision, "output_dir": str(output_dir),
        }
        command = [sys.executable, str(inputs["training_script"]), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "specialist": specialist, "started_at": now(), "command": command}
        state["active"] = active; state["updated_at"] = now(); atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(
                command, cwd=str(inputs["training_script"].resolve().parents[1]),
                stdout=handle, stderr=subprocess.STDOUT, check=False,
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
        if state["failed"]:
            break
    state["status"] = (
        "complete_validation_only"
        if not state["failed"] and len(state["completed"]) == len(candidates)
        else "failed_closed"
    )
    state["updated_at"] = now(); atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}))
    return 0 if state["status"] == "complete_validation_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
