#!/usr/bin/env python3
"""Validate the VCAS model registry without loading model payloads."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")
VALID_STATUSES = {
    "planned",
    "onnx_validated",
    "engine_validated",
    "deployed",
    "blocked",
}
EXPECTED_OUTPUTS = {
    "detection": ["vehicle_detections"],
    "attributes": ["body_type", "color"],
}
CONFIG_MODEL_KEYS = {
    "detection": "vehicle_detector",
    "attributes": "vehicle_attribute",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_relative_path(value: object, suffix: str | None = None) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value.removeprefix("./"))
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        return False
    return suffix is None or value.endswith(suffix)


def _valid_optional_sha(value: object) -> bool:
    return value is None or (isinstance(value, str) and SHA256_RE.fullmatch(value) is not None)


def validate_registry(
    registry: dict[str, Any],
    labels: dict[str, Any],
    runtime_config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    required_root = {
        "schema_version",
        "registry_version",
        "labels_version",
        "created_at",
        "target",
        "artifacts",
    }
    missing = sorted(required_root - set(registry))
    if missing:
        return [f"registry root is missing fields: {missing}"]

    if registry["schema_version"] != "1.0":
        errors.append("schema_version must be 1.0")
    if registry["labels_version"] != labels.get("labels_version"):
        errors.append("registry labels_version must match canonical labels")
    target = registry.get("target")
    if target != {
        "os": "windows",
        "backend": "tensorrt",
        "engine_portability": "target_host_only",
    }:
        errors.append("target must freeze Windows TensorRT target-host-only engines")

    config_registry = runtime_config.get("model_registry_path")
    if config_registry != "./models/manifests/model_registry.v1.json":
        errors.append("runtime config must reference model_registry.v1.json")
    config_version = runtime_config.get("config_version")
    try:
        config_stage = int(str(config_version).removeprefix("vehicle-analytics-m"))
    except ValueError:
        config_stage = -1
    if not str(config_version).startswith("vehicle-analytics-m") or config_stage < 2:
        errors.append("runtime config_version must be vehicle-analytics-m2 or later")
    if runtime_config.get("labels_version") != registry.get("labels_version"):
        errors.append("runtime and model registry labels_version must match")

    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        errors.append("registry must contain exactly detection and attributes artifacts")
        return errors

    ids: set[str] = set()
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        artifact_id = artifact.get("artifact_id")
        role = artifact.get("role")
        status = artifact.get("delivery_status")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append(f"{prefix}.artifact_id is required")
        elif artifact_id in ids:
            errors.append(f"duplicate artifact_id: {artifact_id}")
        else:
            ids.add(artifact_id)
        if role not in EXPECTED_OUTPUTS:
            errors.append(f"{prefix}.role must be detection or attributes")
            continue
        if role in roles:
            errors.append(f"duplicate model role: {role}")
        roles.add(role)
        if status not in VALID_STATUSES:
            errors.append(f"{prefix}.delivery_status is invalid")
        if artifact.get("labels_version") != registry.get("labels_version"):
            errors.append(f"{prefix}.labels_version must match registry")

        model_input = artifact.get("input")
        if not isinstance(model_input, dict):
            errors.append(f"{prefix}.input must be an object")
            continue
        width = model_input.get("width")
        height = model_input.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            errors.append(f"{prefix}.input dimensions must be positive integers")
        if {
            "channels": model_input.get("channels"),
            "layout": model_input.get("layout"),
            "color_order": model_input.get("color_order"),
            "dtype": model_input.get("dtype"),
        } != {
            "channels": 3,
            "layout": "NCHW",
            "color_order": "RGB",
            "dtype": "float32",
        }:
            errors.append(f"{prefix}.input preprocessing contract is invalid")
        if artifact.get("output_names") != EXPECTED_OUTPUTS[role]:
            errors.append(f"{prefix}.output_names do not match role {role}")

        deployment = artifact.get("deployment")
        if not isinstance(deployment, dict):
            errors.append(f"{prefix}.deployment must be an object")
            continue
        if deployment.get("backend") != "tensorrt" or deployment.get("target_os") != "windows":
            errors.append(f"{prefix}.deployment must target TensorRT on Windows")
        if deployment.get("precision") not in {"fp32", "fp16", "int8"}:
            errors.append(f"{prefix}.deployment.precision is invalid")
        max_batch = deployment.get("max_batch")
        if not isinstance(max_batch, int) or isinstance(max_batch, bool) or not 1 <= max_batch <= 128:
            errors.append(f"{prefix}.deployment.max_batch must be in [1, 128]")

        files = artifact.get("files")
        if not isinstance(files, dict):
            errors.append(f"{prefix}.files must be an object")
            continue
        if not _safe_relative_path(files.get("onnx_path"), ".onnx"):
            errors.append(f"{prefix}.files.onnx_path must be a safe relative .onnx path")
        if not _safe_relative_path(files.get("engine_path"), ".engine"):
            errors.append(f"{prefix}.files.engine_path must be a safe relative .engine path")
        onnx_sha = files.get("onnx_sha256")
        engine_sha = files.get("engine_sha256")
        if not _valid_optional_sha(onnx_sha):
            errors.append(f"{prefix}.files.onnx_sha256 is invalid")
        if not _valid_optional_sha(engine_sha):
            errors.append(f"{prefix}.files.engine_sha256 is invalid")

        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{prefix}.provenance must be an object")
            continue
        code_commit = provenance.get("code_commit")
        if code_commit is not None and (
            not isinstance(code_commit, str) or COMMIT_RE.fullmatch(code_commit) is None
        ):
            errors.append(f"{prefix}.provenance.code_commit is invalid")
        for path_field in ("model_card_path", "metrics_path"):
            path_value = provenance.get(path_field)
            if path_value is not None and not _safe_relative_path(path_value):
                errors.append(f"{prefix}.provenance.{path_field} must be a safe relative path")

        if status == "planned":
            if onnx_sha is not None or engine_sha is not None:
                errors.append(f"{prefix} planned artifact must not claim file hashes")
        elif status not in {"blocked", None}:
            required_provenance = (
                "dataset_version",
                "training_run_id",
                "code_commit",
                "model_card_path",
                "metrics_path",
            )
            if onnx_sha is None:
                errors.append(f"{prefix} validated artifact requires onnx_sha256")
            if any(provenance.get(field) in (None, "") for field in required_provenance):
                errors.append(f"{prefix} validated artifact requires complete provenance")
        if status in {"engine_validated", "deployed"} and engine_sha is None:
            errors.append(f"{prefix} engine-validated artifact requires engine_sha256")

        runtime_key = CONFIG_MODEL_KEYS[role]
        runtime_model = runtime_config.get("models", {}).get(runtime_key)
        if not isinstance(runtime_model, dict):
            errors.append(f"runtime config is missing {runtime_key}")
            continue
        comparisons = {
            "artifact": artifact_id,
            "role": role,
            "delivery_status": status,
            "backend": deployment.get("backend"),
            "precision": deployment.get("precision"),
            "onnx_path": f"./{files.get('onnx_path')}",
            "engine_path": f"./{files.get('engine_path')}",
            "input_size": width,
            "max_batch": max_batch,
        }
        for field, expected in comparisons.items():
            if runtime_model.get(field) != expected:
                errors.append(
                    f"runtime {runtime_key}.{field} does not match registry: "
                    f"{runtime_model.get(field)!r} != {expected!r}"
                )

    if roles != set(EXPECTED_OUTPUTS):
        errors.append("registry must contain one detection and one attributes role")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("config/vehicle_labels.v1.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/vehicle_analytics.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        registry = load_json(args.registry)
        labels = load_json(args.labels)
        runtime_config = load_json(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors = validate_registry(registry, labels, runtime_config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: {args.registry} registry_version={registry['registry_version']} "
        f"artifacts={len(registry['artifacts'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
