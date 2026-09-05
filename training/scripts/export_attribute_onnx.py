#!/usr/bin/env python3
"""Export a VCAS attribute checkpoint to ONNX and validate its output contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="optional per-head temperature calibration JSON",
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("install training/requirements-cloud.txt first") from exc

    from src.multitask_mobilenet_v3 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        architecture_from_checkpoint,
        model_from_checkpoint,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    body_types = list(checkpoint["body_types"])
    colors = list(checkpoint["colors"])
    input_size = int(checkpoint["input_size"])
    labels = load_json(args.labels)
    registry = load_json(args.registry)
    expected_body_types = [str(value) for value in labels["body_types"]]
    expected_colors = [str(value) for value in labels["colors"]]
    if body_types != expected_body_types:
        raise RuntimeError(
            "checkpoint body_type order does not match config/vehicle_labels.v1.json"
        )
    if colors != expected_colors:
        raise RuntimeError(
            "checkpoint color order does not match config/vehicle_labels.v1.json"
        )
    attribute_artifact = next(
        artifact
        for artifact in registry["artifacts"]
        if artifact.get("role") == "attributes"
    )
    expected_input = attribute_artifact["input"]
    if input_size != int(expected_input["width"]) or input_size != int(
        expected_input["height"]
    ):
        raise RuntimeError(
            f"checkpoint input size {input_size} does not match model registry "
            f"{expected_input['width']}x{expected_input['height']}"
        )
    expected_output_names = list(attribute_artifact["output_names"])
    if expected_output_names != ["body_type", "color"]:
        raise RuntimeError(
            "attribute registry output_names must be exactly ['body_type', 'color']"
        )
    model = model_from_checkpoint(checkpoint, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    class RuntimeInputModel(nn.Module):
        """Accept the adapter's RGB [0,1] tensor and normalize inside ONNX."""

        def __init__(self, base):
            super().__init__()
            self.base = base
            self.register_buffer(
                "mean",
                torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            )

        def forward(self, images):
            return self.base((images - self.mean) / self.std)

    model = RuntimeInputModel(model)
    model.eval()
    calibration = None
    if args.calibration:
        calibration = load_json(args.calibration)
        if calibration.get("method") != "per_head_temperature_scaling":
            raise RuntimeError("unsupported attribute calibration method")
        body_temperature = float(calibration["body_type"]["temperature"])
        color_temperature = float(calibration["color"]["temperature"])

        class TemperatureScaledModel(nn.Module):
            def __init__(self, base, body_value: float, color_value: float):
                super().__init__()
                self.base = base
                self.body_value = body_value
                self.color_value = color_value

            def forward(self, images):
                body_logits, color_logits = self.base(images)
                return (
                    body_logits / self.body_value,
                    color_logits / self.color_value,
                )

        model = TemperatureScaledModel(
            model, body_temperature, color_temperature
        )
        model.eval()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=expected_output_names,
        dynamic_axes={
            "images": {0: "batch"},
            "body_type": {0: "batch"},
            "color": {0: "batch"},
        },
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    graph_inputs = [value.name for value in onnx_model.graph.input]
    graph_outputs = [value.name for value in onnx_model.graph.output]
    if graph_inputs != ["images"] or graph_outputs != expected_output_names:
        raise RuntimeError(
            f"unexpected ONNX tensor names: inputs={graph_inputs}, "
            f"outputs={graph_outputs}"
        )
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    body_output, color_output = session.run(
        None,
        {"images": np.zeros((2, 3, input_size, input_size), dtype=np.float32)},
    )
    expected_body = (2, len(body_types))
    expected_color = (2, len(colors))
    if body_output.shape != expected_body or color_output.shape != expected_color:
        raise RuntimeError(
            f"unexpected ONNX shapes: body={body_output.shape}, "
            f"color={color_output.shape}"
        )
    summary = {
        "schema_version": "1.0",
        "artifact_id": output.stem,
        "architecture": architecture_from_checkpoint(checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "input": {
            "name": "images",
            "shape": ["batch", 3, input_size, input_size],
            "dtype": "float32",
            "range": "[0,1]",
            "normalization": "embedded_imagenet_mean_std",
        },
        "outputs": {
            "body_type": ["batch", len(body_types)],
            "color": ["batch", len(colors)],
        },
        "body_types": body_types,
        "colors": colors,
        "calibration": calibration,
        "onnx_checker": "pass",
        "onnxruntime_cpu_smoke": "pass",
        "release_eligible": False,
    }
    write_json(output.with_suffix(".onnx.json"), summary)
    print(f"PASS: exported and validated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
