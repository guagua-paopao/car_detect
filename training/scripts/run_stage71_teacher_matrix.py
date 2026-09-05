#!/usr/bin/env python3
"""Run fail-closed Stage71 ConvNeXt body and color teacher training."""

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


def usable_train_rows(rows: list[dict[str, str]], specialist: str) -> list[dict[str, str]]:
    train = [row for row in rows if row.get("split") == "train"]
    if specialist == "body":
        return [row for row in train if (
            (truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown", None})
            or row.get("coarse_body_family") in {"car", "truck"}
        )]
    if specialist == "color":
        return [row for row in train if truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown", None}]
    raise ValueError(f"invalid specialist: {specialist}")


def expected_supervised_quota(rows: list[dict[str, str]], parameters: dict) -> dict[str, float]:
    weighted: list[tuple[float, bool, bool, bool, str]] = []
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
        if small:
            weight *= 1.0 + float(parameters.get("small_sample_weight", 0.0))
        if truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown", None}:
            weight *= 1.0 + float(parameters.get("color_sample_weight", 0.0))
        if night:
            weight *= float(parameters.get("night_sample_weight", 1.0))
        if occluded:
            weight *= float(parameters.get("occlusion_sample_weight", 1.0))
        weighted.append((weight, night, small, occluded, row.get("stage71_origin", "")))
    total = sum(item[0] for item in weighted)
    if total <= 0:
        raise RuntimeError("supervised sampler has zero total weight")
    return {
        "rows": len(rows),
        "night": sum(item[0] for item in weighted if item[1]) / total,
        "small": sum(item[0] for item in weighted if item[2]) / total,
        "occluded_or_truncated": sum(item[0] for item in weighted if item[3]) / total,
        "cctv": sum(item[0] for item in weighted if item[4] == "cctv") / total,
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
        raise FileExistsError("refusing to overwrite Stage71 teacher evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage71 matrix is not prepared")
    if matrix.get("runner_sha256", "").lower() != sha256(Path(__file__).resolve()):
        raise RuntimeError("Stage71 runner SHA256 mismatch")
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
            raise RuntimeError(f"Stage71 execution policy mismatch: {key}")
    if policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage71 lost test isolation")

    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"Stage71 immutable input mismatch: {name}")
        inputs[name] = path
    needed = {"training_script", "labels", "manifest_report", "body_manifest", "color_manifest", "unlabeled_manifest"}
    if not needed.issubset(inputs):
        raise RuntimeError("Stage71 immutable input set is incomplete")
    report = json.loads(inputs["manifest_report"].read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("eligibility") != "research-only_non-deployable":
        raise RuntimeError("Stage71 manifest report did not pass")
    report_policy = report.get("policy", {})
    if (
        report_policy.get("test_rows_imported") is not False
        or report_policy.get("frozen_video_used") is not False
        or report_policy.get("color_teacher_uses_dvm_replay_instead_of_cctv_only_refinetune") is not True
    ):
        raise RuntimeError("Stage71 manifest policy is invalid")
    if report["outputs"]["body_teacher"]["sha256"].lower() != sha256(inputs["body_manifest"]):
        raise RuntimeError("Stage71 report does not bind the body manifest")
    if report["outputs"]["color_teacher"]["sha256"].lower() != sha256(inputs["color_manifest"]):
        raise RuntimeError("Stage71 report does not bind the color manifest")
    if report["outputs"]["color_teacher"]["splits"].get("train", 0) < 118338:
        raise RuntimeError("Stage71 effective color train rows are below 118338")
    if report.get("post_filter_cross_split_near_leaks") or report.get("post_filter_group_leaks"):
        raise RuntimeError("Stage71 manifest retains a split leak")

    safety_root = Path(matrix.get("datasets_safety_root", "")).resolve()
    if not safety_root.is_dir():
        raise RuntimeError("Stage71 datasets safety root is missing")
    manifests = {"body": inputs["body_manifest"], "color": inputs["color_manifest"], "unlabeled": inputs["unlabeled_manifest"]}
    rows = {name: load_rows(path) for name, path in manifests.items()}
    for name, manifest_rows in rows.items():
        for row in manifest_rows:
            raw = Path(row.get("image_path", ""))
            resolved = raw.resolve() if raw.is_absolute() else (manifests[name].parent / raw).resolve()
            try:
                resolved.relative_to(safety_root)
            except ValueError as error:
                raise RuntimeError(f"{name} image escapes the datasets safety root: {resolved}") from error
            if row.get("split") == "test":
                raise RuntimeError(f"{name} manifest contains a test row")

    candidates = list(matrix.get("candidates", []))
    if len(candidates) != 2 or {item.get("specialist") for item in candidates} != {"body", "color"}:
        raise RuntimeError("Stage71 requires exactly one body and one color teacher")
    quota_reports = {}
    for candidate in candidates:
        specialist = candidate["specialist"]
        parameters = {**matrix["common"], **candidate}
        isolated = (
            parameters.get("architecture") == "convnext_tiny"
            and parameters.get("selection_head") == specialist
            and float(parameters.get(f"{specialist if specialist == 'color' else 'body'}_loss_weight", 0)) > 0
            and float(parameters.get("body_loss_weight" if specialist == "color" else "color_loss_weight", -1)) == 0
            and float(parameters.get("unlabeled_body_weight" if specialist == "body" else "unlabeled_color_weight", 0)) > 0
            and float(parameters.get("unlabeled_color_weight" if specialist == "body" else "unlabeled_body_weight", -1)) == 0
        )
        if not isolated:
            raise RuntimeError(f"{candidate.get('candidate_id')} lost ConvNeXt specialist isolation")
        checkpoint = Path(candidate["init_checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint).lower() != candidate["init_checkpoint_sha256"].lower():
            raise RuntimeError(f"Stage71 init checkpoint mismatch: {checkpoint}")
        supervised = expected_supervised_quota(usable_train_rows(rows[specialist], specialist), parameters)
        unlabeled_night = expected_unlabeled_night(rows["unlabeled"], float(parameters.get("unlabeled_night_sample_weight", 1.0)))
        minimum = float(matrix["quota_policy"][f"minimum_{specialist}_expected_night_fraction"])
        if supervised["night"] < minimum or unlabeled_night < float(matrix["quota_policy"]["minimum_unlabeled_expected_night_fraction"]):
            raise RuntimeError(f"{candidate['candidate_id']} misses the sampled night quota")
        if specialist == "color" and supervised["cctv"] < float(matrix["quota_policy"]["minimum_color_expected_cctv_fraction"]):
            raise RuntimeError(f"{candidate['candidate_id']} misses the sampled CCTV quota")
        quota_reports[candidate["candidate_id"]] = {"supervised": supervised, "unlabeled_night": unlabeled_night}
    if args.preflight_only:
        print(json.dumps({"status": "pass_preflight_only", "expected_sampler_quotas": quota_reports, "test_accessed": False, "frozen_video_used": False}, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": "stage71-convnext-teachers-run-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "expected_sampler_quotas": quota_reports,
        "candidates_total": len(candidates),
        "completed": [],
        "failed": [],
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "eligibility": "research-only_non-deployable",
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.state, state)
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        specialist = candidate["specialist"]
        output_dir = args.output_root / candidate_id
        parameters = {
            **matrix["common"],
            **candidate,
            "manifest": str(manifests[specialist].resolve()),
            "unlabeled_manifest": str(manifests["unlabeled"].resolve()),
            "unlabeled_root": str(safety_root),
            "labels": str(inputs["labels"].resolve()),
            "device": args.device,
            "dataset_version": "attribute-domain-v2-stage71-convnext-teachers-research-v1",
            "code_revision": args.code_revision,
            "output_dir": str(output_dir),
        }
        command = [sys.executable, str(inputs["training_script"]), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {"candidate_id": candidate_id, "specialist": specialist, "started_at": now(), "command": command}
        state["active"] = active
        state["updated_at"] = now()
        atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(command, cwd=str(inputs["training_script"].resolve().parents[1]), stdout=handle, stderr=subprocess.STDOUT, check=False)
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
        if state["failed"]:
            break
    state["status"] = "complete_validation_only" if not state["failed"] and len(state["completed"]) == 2 else "failed_closed"
    state["updated_at"] = now()
    atomic_json(args.state, state)
    print(json.dumps({"status": state["status"], "completed": len(state["completed"]), "failed": len(state["failed"])}))
    return 0 if state["status"] == "complete_validation_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
