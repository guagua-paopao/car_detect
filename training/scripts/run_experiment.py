#!/usr/bin/env python3
"""Run one versioned VCAS experiment from a checked-in JSON configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


DETECTION_KEYS = {
    "experiment_id",
    "model",
    "imgsz",
    "epochs",
    "patience",
    "batch",
    "workers",
    "amp",
    "seed",
    "deterministic",
    "save_period",
}
ATTRIBUTE_KEYS = {
    "experiment_id",
    "architecture",
    "input_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "color_loss_weight",
    "class_weighting",
    "label_smoothing",
    "gradient_clip_norm",
    "patience",
    "type_threshold",
    "color_threshold",
    "workers",
    "seed",
    "amp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json")
    parser.add_argument(
        "--attribute-manifest",
        type=Path,
        help="override dataset-root/attributes/attribute_manifest.csv",
    )
    parser.add_argument("--dataset-version", default="dataset-v1")
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--device", default="0")
    parser.add_argument("--code-revision", default=os.environ.get("VCAS_CODE_REVISION", "unknown"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="skip completed runs and resume incomplete runs when last.pt exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_config(config: dict[str, Any]) -> str:
    if "model" in config:
        kind = "detection"
        allowed = DETECTION_KEYS
    elif "architecture" in config:
        kind = "attributes"
        allowed = ATTRIBUTE_KEYS
    else:
        raise ValueError("configuration must describe a detection or attribute experiment")
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unsupported {kind} configuration keys: {sorted(unknown)}")
    missing = {"experiment_id", "epochs", "seed"} - set(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    if config.get("amp") is not True or config.get("deterministic", True) is not True:
        raise ValueError("VCAS formal experiments require amp=true and deterministic=true")
    return kind


def code_revision(args: argparse.Namespace) -> str:
    if args.code_revision != "unknown":
        return args.code_revision
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if value:
            return value
    except (OSError, subprocess.CalledProcessError):
        pass
    return "unknown"


def build_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    kind: str,
    revision: str,
) -> tuple[list[str], Path, Path | None]:
    experiment_id = str(config["experiment_id"])
    common = [
        "--run-kind",
        args.run_kind,
        "--dataset-version",
        args.dataset_version,
        "--code-revision",
        revision,
    ]
    if kind == "detection":
        run_dir = args.runs_root.resolve() / experiment_id
        checkpoint = run_dir / "weights" / "last.pt"
        data_yaml = args.dataset_root.resolve() / "detection" / (
            "vehicle_det_v1.yaml"
            if args.dataset_version != "dataset-pilot-v1"
            else "vehicle_det_pilot.yaml"
        )
        command = [
            sys.executable,
            str(TRAINING_ROOT / "scripts" / "train_detector.py"),
            "--data",
            str(data_yaml),
            "--model",
            str(config["model"]),
            "--imgsz",
            str(config["imgsz"]),
            "--epochs",
            str(config["epochs"]),
            "--batch",
            str(config["batch"]),
            "--device",
            args.device,
            "--workers",
            str(config["workers"]),
            "--patience",
            str(config["patience"]),
            "--seed",
            str(config["seed"]),
            "--project",
            str(args.runs_root.resolve()),
            "--name",
            experiment_id,
            "--save-period",
            str(config["save_period"]),
            *common,
        ]
        resume_requested = args.resume or (args.auto_resume and checkpoint.is_file())
        if resume_requested:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint}")
            command.extend(["--resume", str(checkpoint)])
        return command, run_dir, None

    run_dir = args.runs_root.resolve() / experiment_id
    checkpoint = run_dir / "last.pt"
    manifest = (
        args.attribute_manifest.resolve()
        if args.attribute_manifest
        else args.dataset_root.resolve() / "attributes" / "attribute_manifest.csv"
    )
    command = [
        sys.executable,
        str(TRAINING_ROOT / "scripts" / "train_attribute.py"),
        "--manifest",
        str(manifest),
        "--labels",
        str(args.labels.resolve()),
        "--input-size",
        str(config["input_size"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--workers",
        str(config["workers"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--color-loss-weight",
        str(config["color_loss_weight"]),
        "--class-weighting",
        str(config["class_weighting"]),
        "--label-smoothing",
        str(config["label_smoothing"]),
        "--gradient-clip-norm",
        str(config["gradient_clip_norm"]),
        "--patience",
        str(config["patience"]),
        "--seed",
        str(config["seed"]),
        "--device",
        "cuda" if args.device == "0" else args.device,
        "--output-dir",
        str(run_dir),
        *common,
    ]
    if "type_threshold" in config:
        command.extend(["--type-threshold", str(config["type_threshold"])])
    if "color_threshold" in config:
        command.extend(["--color-threshold", str(config["color_threshold"])])
    resume_requested = args.resume or (args.auto_resume and checkpoint.is_file())
    if resume_requested:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint}")
        command.extend(["--resume", str(checkpoint)])
    output = args.artifacts_root.resolve() / f"vehicle-attr-{experiment_id.lower()}.onnx"
    return command, run_dir, output


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    kind = check_config(config)
    revision = code_revision(args)
    if args.run_kind == "formal" and revision == "unknown":
        raise RuntimeError(
            "formal runs require --code-revision or VCAS_CODE_REVISION; "
            "use the Git commit or source-package SHA256"
        )

    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    args.manifests_root.mkdir(parents=True, exist_ok=True)
    experiment_id = str(config["experiment_id"])
    record_path = args.manifests_root.resolve() / f"{experiment_id}.experiment.json"
    if args.auto_resume and record_path.is_file():
        existing = load_json(record_path)
        existing_onnx = Path(str(existing.get("onnx", "")))
        if existing.get("status") == "completed" and existing_onnx.is_file():
            print(f"SKIP: experiment {experiment_id} is already complete: {record_path}")
            return 0

    command, run_dir, attribute_onnx = build_command(args, config, kind, revision)
    will_resume = "--resume" in command
    if run_dir.exists() and not will_resume and not args.dry_run:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use --resume or choose a new experiment id"
        )

    record = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "kind": kind,
        "status": "dry_run" if args.dry_run else "running",
        "started_at": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_version": args.dataset_version,
        "code_revision": revision,
        "command": command,
    }
    write_json(record_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if kind == "detection":
            source_onnx = run_dir / "weights" / "best.onnx"
            if not source_onnx.is_file():
                raise FileNotFoundError(f"detection ONNX is missing: {source_onnx}")
            target_onnx = args.artifacts_root.resolve() / f"vehicle-det-{experiment_id.lower()}.onnx"
            shutil.copy2(source_onnx, target_onnx)
            record["onnx"] = str(target_onnx)
            record["onnx_sha256"] = sha256_file(target_onnx)
        else:
            assert attribute_onnx is not None
            subprocess.run(
                [
                    sys.executable,
                    str(TRAINING_ROOT / "scripts" / "export_attribute_onnx.py"),
                    "--checkpoint",
                    str(run_dir / "best.pt"),
                    "--output",
                    str(attribute_onnx),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            record["onnx"] = str(attribute_onnx)
            record["onnx_sha256"] = sha256_file(attribute_onnx)
    except Exception as exc:
        record["status"] = "failed"
        record["finished_at"] = utc_now()
        record["error"] = f"{type(exc).__name__}: {exc}"
        write_json(record_path, record)
        raise

    record["status"] = "completed"
    record["finished_at"] = utc_now()
    record["run_dir"] = str(run_dir)
    write_json(record_path, record)
    print(f"PASS: experiment {experiment_id} completed; record={record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
