#!/usr/bin/env python3
"""Run the isolated Stage76 DVM body pretrain followed by real-CCTV fine-tuning."""

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
    "candidate_id", "purpose", "phase", "manifest_key", "init_checkpoint_sha256",
    "use_unlabeled_stream",
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


def validate_manifest_paths(
    manifest: Path,
    rows: list[dict[str, str]],
    safety_root: Path,
    *,
    unlabeled: bool,
) -> dict[str, int]:
    counters = {"rows": 0, "train": 0, "validation": 0, "body_supervised": 0}
    for row in rows:
        split = row.get("split")
        if split not in ({"train"} if unlabeled else {"train", "validation"}):
            raise RuntimeError(f"invalid split in {manifest}: {split}")
        if "vcas_rtsp_demo_60s" in str(row).lower() or "36-48" in str(row).lower():
            raise RuntimeError(f"frozen-video marker in {manifest}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"image escapes dataset safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest image missing: {resolved}")
        if unlabeled and (
            truthy(row.get("body_type_supervised"))
            or truthy(row.get("color_supervised"))
            or row.get("body_type") not in {"", "unknown", None}
            or row.get("color") not in {"", "unknown", None}
        ):
            raise RuntimeError("unlabeled stream contains an attribute target")
        counters["rows"] += 1
        counters[split] += 1
        if truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown", None}:
            counters["body_supervised"] += 1
    return counters


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
        raise FileExistsError("refusing to overwrite Stage76 training evidence")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_validation_only":
        raise RuntimeError("Stage76 matrix is not prepared")
    if matrix.get("runner_sha256", "").lower() != sha256(Path(__file__).resolve()):
        raise RuntimeError("Stage76 runner SHA256 mismatch")
    policy = matrix.get("execution_policy", {})
    for key, expected in {
        "iterative_test_access": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
        "research_only": True,
    }.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"Stage76 execution policy mismatch: {key}")
    if policy.get("required_flag") != "--skip-test":
        raise RuntimeError("Stage76 lost test isolation")

    inputs: dict[str, Path] = {}
    for name, item in matrix.get("immutable_inputs", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path).lower() != item["sha256"].lower():
            raise RuntimeError(f"immutable input mismatch: {name}")
        inputs[name] = path
    needed = {
        "training_script", "labels", "dvm_manifest", "dvm_report", "domain_manifest",
        "domain_report", "unlabeled_manifest", "init_checkpoint",
    }
    if not needed.issubset(inputs):
        raise RuntimeError("Stage76 immutable input set is incomplete")

    dvm_report = json.loads(inputs["dvm_report"].read_text(encoding="utf-8"))
    if (
        dvm_report.get("status") != "pass"
        or dvm_report.get("output", {}).get("sha256", "").lower() != sha256(inputs["dvm_manifest"])
        or dvm_report.get("policy", {}).get("test_accessed") is not False
        or dvm_report.get("policy", {}).get("frozen_video_used") is not False
    ):
        raise RuntimeError("DVM body manifest report is not training-eligible")
    domain_report = json.loads(inputs["domain_report"].read_text(encoding="utf-8"))
    integrity = domain_report.get("integrity", {})
    if (
        domain_report.get("status") != "pass"
        or domain_report.get("output", {}).get("sha256", "").lower() != sha256(inputs["domain_manifest"])
        or integrity.get("group_leaks") != 0
        or integrity.get("exact_image_leaks") != 0
        or integrity.get("near_duplicate_leaks") != 0
        or integrity.get("test_rows") != 0
        or integrity.get("frozen_markers") != 0
    ):
        raise RuntimeError("Stage72 body domain manifest report is not training-eligible")

    safety_root = Path(matrix["datasets_safety_root"]).resolve()
    if not safety_root.is_dir():
        raise RuntimeError("dataset safety root is missing")
    rows = {
        "dvm_manifest": load_rows(inputs["dvm_manifest"]),
        "domain_manifest": load_rows(inputs["domain_manifest"]),
        "unlabeled_manifest": load_rows(inputs["unlabeled_manifest"]),
    }
    audits = {
        "dvm": validate_manifest_paths(inputs["dvm_manifest"], rows["dvm_manifest"], safety_root, unlabeled=False),
        "domain": validate_manifest_paths(inputs["domain_manifest"], rows["domain_manifest"], safety_root, unlabeled=False),
        "unlabeled": validate_manifest_paths(inputs["unlabeled_manifest"], rows["unlabeled_manifest"], safety_root, unlabeled=True),
    }
    if audits["dvm"]["body_supervised"] < 100000 or audits["domain"]["body_supervised"] < 100000:
        raise RuntimeError("Stage76 supervised body support fell below the fixed minimum")

    phases = matrix.get("phases", [])
    if [phase.get("phase") for phase in phases] != ["dvm_pretrain", "cctv_finetune"]:
        raise RuntimeError("Stage76 requires DVM pretrain followed by CCTV fine-tune")
    if args.preflight_only:
        print(json.dumps({
            "status": "pass_preflight_only", "audits": audits,
            "test_accessed": False, "frozen_video_used": False,
        }, ensure_ascii=False, indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": "stage76-body-pretrain-finetune-run-v1",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "manifest_audits": audits,
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
    previous_checkpoint = inputs["init_checkpoint"]
    for phase in phases:
        candidate_id = phase["candidate_id"]
        manifest = inputs[phase["manifest_key"]]
        output_dir = args.output_root / candidate_id
        parameters = {
            **matrix["common"],
            **phase,
            "manifest": str(manifest.resolve()),
            "labels": str(inputs["labels"].resolve()),
            "init_checkpoint": str(previous_checkpoint.resolve()),
            "device": args.device,
            "code_revision": args.code_revision,
            "output_dir": str(output_dir),
        }
        if phase.get("use_unlabeled_stream"):
            parameters.update({
                "unlabeled_manifest": str(inputs["unlabeled_manifest"].resolve()),
                "unlabeled_root": str(safety_root),
            })
        command = [sys.executable, str(inputs["training_script"].resolve()), *command_arguments(parameters)]
        if "--skip-test" not in command:
            raise RuntimeError(f"{candidate_id} lost --skip-test")
        active = {
            "candidate_id": candidate_id,
            "phase": phase["phase"],
            "started_at": now(),
            "init_checkpoint": str(previous_checkpoint.resolve()),
            "init_checkpoint_sha256": sha256(previous_checkpoint),
            "command": command,
        }
        state["active"] = active
        state["updated_at"] = now()
        atomic_json(args.state, state)
        log = args.output_root / f"{candidate_id}.stdout.log"
        with log.open("wb") as handle:
            result = subprocess.run(
                command,
                cwd=str(inputs["training_script"].resolve().parents[1]),
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
        if state["failed"]:
            break
        previous_checkpoint = best
    state["status"] = (
        "complete_validation_only_two_phase"
        if not state["failed"] and len(state["completed"]) == 2
        else "failed_closed"
    )
    state["updated_at"] = now()
    atomic_json(args.state, state)
    print(json.dumps({
        "status": state["status"], "completed": len(state["completed"]),
        "failed": len(state["failed"]),
    }))
    return 0 if state["status"] == "complete_validation_only_two_phase" else 2


if __name__ == "__main__":
    raise SystemExit(main())
