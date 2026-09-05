#!/usr/bin/env python3
"""Queue the taxonomy-v2 body candidate after the Stage76/77/80 GPU chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


META_KEYS = {"candidate_id", "purpose"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def command_arguments(parameters: dict[str, Any]) -> list[str]:
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


def stage76_action(status: str) -> str:
    if status == "running":
        return "wait"
    if status == "complete_validation_only_two_phase":
        return "ready"
    return "fail"


def stage80_action(status: str) -> str:
    if status in {"waiting_for_stage77", "running"}:
        return "wait"
    if status in {"complete_validation_only", "failed_closed_training", "failed_closed_runtime"}:
        return "ready"
    return "fail"


def select_stage76_checkpoint(state: dict[str, Any]) -> tuple[Path, str]:
    if state.get("status") != "complete_validation_only_two_phase":
        raise RuntimeError("Stage76 is not complete")
    records = [
        item for item in state.get("completed", [])
        if item.get("phase") == "cctv_finetune"
    ]
    if len(records) != 1:
        raise RuntimeError("Stage76 must contain exactly one CCTV fine-tune record")
    record = records[0]
    checkpoint = Path(str(record.get("best_checkpoint", "")))
    expected = str(record.get("best_checkpoint_sha256", "")).lower()
    if not checkpoint.is_file() or not expected or sha256(checkpoint).lower() != expected:
        raise RuntimeError("Stage76 best checkpoint lineage mismatch")
    return checkpoint, expected


def validate_body_manifest(
    manifest: Path,
    report: dict[str, Any],
    labels: dict[str, Any],
    safety_root: Path,
) -> dict[str, Any]:
    if report.get("status") != "pass":
        raise RuntimeError("Stage83 V5 report did not pass")
    output = report.get("output", {})
    if str(output.get("manifest_sha256", "")).lower() != sha256(manifest).lower():
        raise RuntimeError("Stage83 manifest/report SHA256 mismatch")
    integrity = report.get("integrity", {})
    for key in (
        "stage82_to_validation_exact_leaks", "stage82_to_validation_near_leaks",
        "stage82_to_validation_group_leaks", "test_rows", "frozen_markers",
    ):
        if integrity.get(key) != 0:
            raise RuntimeError(f"Stage83 integrity violation: {key}")
    policy = report.get("policy", {})
    for key in (
        "stage82_is_train_only", "stage72_validation_preserved",
        "stage82_review_status_normalized_for_loader",
        "stage82_research_only_scope_preserved",
        "pickup_exact_truth_excluded_from_coarse_truck_loss",
    ):
        if policy.get(key) is not True:
            raise RuntimeError(f"Stage83 policy violation: {key}")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if policy.get(key) is not False:
            raise RuntimeError(f"Stage83 isolation violation: {key}")

    allowed_body = set(labels["body_types"])
    counts: Counter[str] = Counter()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            split = row.get("split", "").strip().lower()
            if split not in {"train", "validation"}:
                raise RuntimeError(f"invalid Stage83 split at line {line}: {split}")
            searchable = " ".join(row.values()).lower()
            if "vcas_rtsp_demo_60s" in searchable or "36-48" in searchable:
                raise RuntimeError(f"frozen marker at Stage83 line {line}")
            raw = Path(row.get("image_path", ""))
            resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
            try:
                resolved.relative_to(safety_root)
            except ValueError as exc:
                raise RuntimeError(f"Stage83 image escapes safety root: {resolved}") from exc
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            counts[f"split:{split}"] += 1
            supervised = truthy(row.get("body_type_supervised")) and row.get("body_type") in allowed_body
            if split == "train" and row.get("review_status") == "approved" and supervised:
                counts["loader_usable_supervised_train"] += 1
            if row.get("stage83_origin") == "stage82_mio":
                counts["stage82_rows"] += 1
                if (
                    split != "train"
                    or row.get("review_status") != "approved"
                    or not supervised
                    or not truthy(row.get("research_only"))
                    or truthy(row.get("deployment_eligible"))
                ):
                    raise RuntimeError(f"Stage82 loader/license contract violation at line {line}")
                counts["stage82_loader_usable"] += 1
                if row.get("body_type") == "pickup" and row.get("coarse_body_family", "").strip():
                    raise RuntimeError(f"pickup coarse-truck conflict at line {line}")
    if counts["stage82_rows"] != 33817 or counts["stage82_loader_usable"] != 33817:
        raise RuntimeError(f"Stage82 usable count mismatch: {dict(counts)}")
    if counts["loader_usable_supervised_train"] != 153881:
        raise RuntimeError(f"Stage83 supervised count mismatch: {dict(counts)}")
    return dict(counts)


def validate_matrix(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    matrix = read_json(args.matrix)
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage84 matrix is not prepared")
    if str(matrix.get("runner_sha256", "")).lower() != sha256(Path(__file__).resolve()).lower():
        raise RuntimeError("Stage84 runner SHA256 mismatch")
    execution = matrix.get("execution_policy", {})
    required = {
        "iterative_test_access": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "research_only": True,
    }
    for key, expected in required.items():
        if execution.get(key) is not expected:
            raise RuntimeError(f"Stage84 execution policy violation: {key}")
    if execution.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage84 lost --skip-test")

    paths: dict[str, Path] = {}
    for name, evidence in matrix.get("immutable_inputs", {}).items():
        path = Path(evidence["path"])
        if not path.is_file() or sha256(path).lower() != str(evidence["sha256"]).lower():
            raise RuntimeError(f"Stage84 immutable input mismatch: {name}")
        paths[name] = path
    labels = read_json(paths["labels"])
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("Stage84 requires taxonomy-v2 labels")
    stage83_state = read_json(paths["manifest_state"])
    stage83_report = read_json(paths["manifest_report"])
    if stage83_state.get("status") != "complete_data_only":
        raise RuntimeError("Stage83 V5 state is not complete")
    if str(stage83_state.get("output_manifest_sha256", "")).lower() != sha256(paths["manifest"]):
        raise RuntimeError("Stage83 state/manifest mismatch")
    if str(stage83_state.get("output_report_sha256", "")).lower() != sha256(paths["manifest_report"]):
        raise RuntimeError("Stage83 state/report mismatch")
    safety_root = Path(matrix["datasets_safety_root"]).resolve()
    if not safety_root.is_dir():
        raise RuntimeError("dataset safety root is missing")
    audits = {
        "body": validate_body_manifest(paths["manifest"], stage83_report, labels, safety_root),
    }
    unlabeled_report = read_json(paths["unlabeled_report"])
    unlabeled = unlabeled_report.get("outputs", {}).get("unlabeled", {})
    if (
        unlabeled_report.get("status") != "pass"
        or str(unlabeled.get("sha256", "")).lower() != sha256(paths["unlabeled_manifest"])
        or unlabeled.get("splits") != {"train": 188072}
        or unlabeled_report.get("policy", {}).get("unlabeled_targets_all_unknown") is not True
        or unlabeled_report.get("policy", {}).get("test_rows_imported") is not False
        or unlabeled_report.get("policy", {}).get("frozen_video_used") is not False
    ):
        raise RuntimeError("Stage84 unlabeled lineage is not eligible")
    audits["unlabeled"] = {"rows": 188072, "train": 188072}
    return matrix, paths, audits


def wait_for_dependencies(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        stage76 = read_json(args.stage76_state)
        stage80 = read_json(args.stage80_state)
        action76 = stage76_action(str(stage76.get("status", "missing")))
        action80 = stage80_action(str(stage80.get("status", "missing")))
        if action76 == "fail":
            raise RuntimeError(f"Stage76 failed: {stage76.get('status')}")
        if action80 == "fail":
            raise RuntimeError(f"Stage80 has an unexpected state: {stage80.get('status')}")
        if action76 == "ready" and action80 == "ready":
            time.sleep(5)
            return stage76, stage80
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage84 dependency wait timed out")
        time.sleep(max(5, args.poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--stage76-state", type=Path, required=True)
    parser.add_argument("--stage80-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists() or args.state.exists():
        raise FileExistsError("refusing to overwrite Stage84 evidence")
    matrix, paths, audits = validate_matrix(args)
    for dependency in (args.stage76_state, args.stage80_state):
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only", "audits": audits,
            "stage76_status": read_json(args.stage76_state).get("status"),
            "stage80_status": read_json(args.stage80_state).get("status"),
            "test_accessed": False, "frozen_video_used": False,
        }, ensure_ascii=False, indent=2))
        return 0

    state: dict[str, Any] = {
        "schema_version": "stage84-v2-body-run-v1",
        "status": "waiting_for_stage76_stage80",
        "created_at": now(), "updated_at": now(),
        "matrix": str(args.matrix.resolve()), "matrix_sha256": sha256(args.matrix),
        "manifest_audits": audits,
        "test_accessed": False, "frozen_video_used": False,
        "production_model_modified": False, "backend_gates_run": False,
        "deployment_performed": False, "deployment_paused_by_user": True,
        "eligibility": "research-only_non-deployable",
    }
    atomic_json(args.state, state)
    try:
        stage76, stage80 = wait_for_dependencies(args)
        # Revalidate every fixed file after the potentially long wait.
        matrix, paths, audits = validate_matrix(args)
        source_checkpoint, source_sha = select_stage76_checkpoint(stage76)
        args.output_root.mkdir(parents=True, exist_ok=False)
        migrated = args.output_root / "stage76-v2-init.pt"
        migration_report = args.output_root / "stage76-v2-migration-report.json"
        migration_log = args.output_root / "stage84-migration.stdout.log"
        migration_command = [
            sys.executable, str(paths["migrator"]),
            "--input-checkpoint", str(source_checkpoint),
            "--labels", str(paths["labels"]),
            "--output-checkpoint", str(migrated),
            "--output-report", str(migration_report),
        ]
        state.update({
            "status": "running_migration", "updated_at": now(),
            "stage76_checkpoint": str(source_checkpoint),
            "stage76_checkpoint_sha256": source_sha,
            "stage80_terminal_status": stage80.get("status"),
            "migration_command": migration_command,
        })
        atomic_json(args.state, state)
        with migration_log.open("wb") as handle:
            migrated_result = subprocess.run(
                migration_command, cwd=str(paths["migrator"].resolve().parents[1]),
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if migrated_result.returncode != 0 or not migrated.is_file() or not migration_report.is_file():
            raise RuntimeError("Stage84 taxonomy migration failed")
        migration = read_json(migration_report)
        if (
            migration.get("status") != "pass"
            or str(migration.get("input_checkpoint_sha256", "")).lower() != source_sha
            or str(migration.get("output_checkpoint_sha256", "")).lower() != sha256(migrated)
        ):
            raise RuntimeError("Stage84 migration evidence mismatch")

        candidate_dir = args.output_root / matrix["parameters"]["candidate_id"]
        parameters = dict(matrix["parameters"])
        parameters.update({
            "manifest": str(paths["manifest"].resolve()),
            "labels": str(paths["labels"].resolve()),
            "init_checkpoint": str(migrated.resolve()),
            "unlabeled_manifest": str(paths["unlabeled_manifest"].resolve()),
            "unlabeled_root": str(Path(matrix["datasets_safety_root"]).resolve()),
            "device": args.device,
            "code_revision": args.code_revision,
            "output_dir": str(candidate_dir.resolve()),
        })
        command = [sys.executable, str(paths["training_script"]), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError("Stage84 training command lost --skip-test")
        state.update({
            "status": "running_training", "updated_at": now(),
            "migration_checkpoint": str(migrated.resolve()),
            "migration_checkpoint_sha256": sha256(migrated),
            "migration_report": str(migration_report.resolve()),
            "migration_report_sha256": sha256(migration_report),
            "command": command,
        })
        atomic_json(args.state, state)
        training_log = args.output_root / "stage84-v2-body.stdout.log"
        with training_log.open("wb") as handle:
            result = subprocess.run(
                command, cwd=str(paths["training_script"].resolve().parents[1]),
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        best = candidate_dir / "best.pt"
        state.update({
            "status": "complete_validation_only" if result.returncode == 0 and best.is_file() else "failed_closed_training",
            "updated_at": now(), "return_code": result.returncode,
            "stdout_log": str(training_log.resolve()), "stdout_log_sha256": sha256(training_log),
            "best_checkpoint": str(best.resolve()) if best.is_file() else None,
            "best_checkpoint_sha256": sha256(best) if best.is_file() else None,
        })
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state["status"] == "complete_validation_only" else 2
    except Exception as error:
        state.update({
            "status": "failed_closed_runtime", "updated_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        atomic_json(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
