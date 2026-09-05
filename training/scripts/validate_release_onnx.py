#!/usr/bin/env python3
"""Validate both release ONNX files against the local VCAS runtime contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-onnx", type=Path, required=True)
    parser.add_argument("--attribute-onnx", type=Path, required=True)
    parser.add_argument("--detection-yaml", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "models" / "manifests" / "model_registry.v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dimensions(value_info: Any) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(str(dimension.dim_param))
        else:
            result.append(None)
    return result


def artifact_for(registry: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in registry["artifacts"]
        if artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"model registry must contain one {role!r} artifact")
    return matches[0]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for release contract validation") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a YAML object")
    return value


def main() -> int:
    args = parse_args()
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required for release contract validation") from exc

    labels = load_json(args.labels)
    registry = load_json(args.registry)
    detector_registry = artifact_for(registry, "detection")
    attribute_registry = artifact_for(registry, "attributes")
    detection_yaml = load_yaml(args.detection_yaml)
    yaml_names = detection_yaml.get("names")
    if isinstance(yaml_names, dict):
        ordered_names = [
            str(yaml_names[index] if index in yaml_names else yaml_names[str(index)])
            for index in range(len(yaml_names))
        ]
    elif isinstance(yaml_names, list):
        ordered_names = [str(value) for value in yaml_names]
    else:
        raise RuntimeError("detection YAML names must be a list or numeric-key mapping")
    expected_vehicle_classes = [str(value) for value in labels["vehicle_classes"]]
    if ordered_names != expected_vehicle_classes:
        raise RuntimeError(
            f"detection YAML names {ordered_names} do not match canonical order "
            f"{expected_vehicle_classes}"
        )

    detection_model = onnx.load(str(args.detection_onnx.resolve()))
    onnx.checker.check_model(detection_model)
    detection_inputs = {
        value.name: dimensions(value) for value in detection_model.graph.input
    }
    detection_outputs = {
        value.name: dimensions(value) for value in detection_model.graph.output
    }
    expected_det_size = int(detector_registry["input"]["width"])
    expected_det_input = [1, 3, expected_det_size, expected_det_size]
    if detection_inputs != {"images": expected_det_input}:
        raise RuntimeError(
            f"detection ONNX inputs must be {{'images': {expected_det_input}}}, "
            f"got {detection_inputs}"
        )
    if list(detection_outputs) != ["output0"]:
        raise RuntimeError(
            f"detection physical output must be ['output0'], got "
            f"{list(detection_outputs)}"
        )
    detection_shape = detection_outputs["output0"]
    feature_count = 4 + len(expected_vehicle_classes)
    if len(detection_shape) != 3 or feature_count not in detection_shape:
        raise RuntimeError(
            f"detection output must contain {feature_count} features for "
            f"4 box values + {len(expected_vehicle_classes)} classes; "
            f"got {detection_shape}"
        )

    attribute_model = onnx.load(str(args.attribute_onnx.resolve()))
    onnx.checker.check_model(attribute_model)
    attribute_inputs = {
        value.name: dimensions(value) for value in attribute_model.graph.input
    }
    attribute_outputs = {
        value.name: dimensions(value) for value in attribute_model.graph.output
    }
    expected_attr_size = int(attribute_registry["input"]["width"])
    attr_input_shape = attribute_inputs.get("images")
    if (
        attr_input_shape is None
        or len(attr_input_shape) != 4
        or attr_input_shape[1:] != [3, expected_attr_size, expected_attr_size]
    ):
        raise RuntimeError(
            "attribute ONNX input must be images:[batch,3,"
            f"{expected_attr_size},{expected_attr_size}], got {attribute_inputs}"
        )
    expected_attr_names = list(attribute_registry["output_names"])
    if list(attribute_outputs) != expected_attr_names:
        raise RuntimeError(
            f"attribute outputs must be {expected_attr_names}, got "
            f"{list(attribute_outputs)}"
        )
    expected_body_count = len(labels["body_types"])
    expected_color_count = len(labels["colors"])
    if attribute_outputs["body_type"][-1] != expected_body_count:
        raise RuntimeError(
            f"body_type output must contain {expected_body_count} logits, "
            f"got {attribute_outputs['body_type']}"
        )
    if attribute_outputs["color"][-1] != expected_color_count:
        raise RuntimeError(
            f"color output must contain {expected_color_count} logits, "
            f"got {attribute_outputs['color']}"
        )

    report = {
        "schema_version": "1.0",
        "status": "pass",
        "labels_version": labels["labels_version"],
        "detection": {
            "path": str(args.detection_onnx.resolve()),
            "sha256": sha256_file(args.detection_onnx.resolve()),
            "input": detection_inputs,
            "outputs": detection_outputs,
            "vehicle_classes": expected_vehicle_classes,
        },
        "attributes": {
            "path": str(args.attribute_onnx.resolve()),
            "sha256": sha256_file(args.attribute_onnx.resolve()),
            "input": attribute_inputs,
            "outputs": attribute_outputs,
            "body_types": list(labels["body_types"]),
            "colors": list(labels["colors"]),
        },
    }
    write_json(args.output.resolve(), report)
    print(f"PASS: both release ONNX contracts validated: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
