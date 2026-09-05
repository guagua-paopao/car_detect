#!/usr/bin/env python3
"""Contract and negative tests for the M2 vehicle model registry."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_model_registry import validate_registry  # noqa: E402


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain an object")
    return value


def require_error(errors: list[str], text: str) -> None:
    if not any(text in error for error in errors):
        raise AssertionError(f"expected error containing {text!r}, got {errors!r}")


def main() -> None:
    registry = load(ROOT / "models" / "manifests" / "model_registry.v1.json")
    labels = load(ROOT / "config" / "vehicle_labels.v1.json")
    runtime = load(ROOT / "config" / "vehicle_analytics.yaml")
    schema = load(ROOT / "api" / "schemas" / "model_registry.v1.schema.json")

    errors = validate_registry(registry, labels, runtime)
    if errors:
        raise AssertionError(f"model registry must pass: {errors!r}")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise AssertionError("model registry schema must use JSON Schema 2020-12")
    if schema.get("additionalProperties") is not False:
        raise AssertionError("model registry schema must reject unknown root fields")

    duplicate_role = copy.deepcopy(registry)
    duplicate_role["artifacts"][1]["role"] = "detection"
    require_error(validate_registry(duplicate_role, labels, runtime), "duplicate model role")

    unsafe_path = copy.deepcopy(registry)
    unsafe_path["artifacts"][0]["files"]["engine_path"] = "../private/model.engine"
    require_error(validate_registry(unsafe_path, labels, runtime), "safe relative .engine path")

    false_deployment = copy.deepcopy(registry)
    false_deployment["artifacts"][1]["delivery_status"] = "deployed"
    false_deployment["artifacts"][1]["files"]["onnx_sha256"] = None
    false_deployment["artifacts"][1]["files"]["engine_sha256"] = None
    for field in (
        "dataset_version",
        "training_run_id",
        "code_commit",
        "model_card_path",
        "metrics_path",
    ):
        false_deployment["artifacts"][1]["provenance"][field] = None
    require_error(validate_registry(false_deployment, labels, runtime), "requires onnx_sha256")
    require_error(validate_registry(false_deployment, labels, runtime), "complete provenance")
    require_error(validate_registry(false_deployment, labels, runtime), "requires engine_sha256")

    bad_output = copy.deepcopy(registry)
    bad_output["artifacts"][1]["output_names"] = ["body_type"]
    require_error(validate_registry(bad_output, labels, runtime), "output_names")

    drifted_runtime = copy.deepcopy(runtime)
    drifted_runtime["models"]["vehicle_attribute"]["input_size"] = 224
    require_error(validate_registry(registry, labels, drifted_runtime), "input_size")

    header = (ROOT / "include" / "server" / "vehicle_model_contract.h").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "IVehicleDetectionRunner",
        "IVehicleAttributeRunner",
        "VehicleModelRegistry",
        "run_generation",
        "crop_sequence",
        "labels_version",
        "artifact_id",
    ):
        if symbol not in header:
            raise AssertionError(f"C++ model contract is missing {symbol}")

    print("PASS: M2 model registry, runner interfaces, and deployment gates")


if __name__ == "__main__":
    main()
