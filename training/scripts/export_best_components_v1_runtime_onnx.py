#!/usr/bin/env python3
"""Export the best independently validated body/color components as one v1 runtime ONNX.

The source checkpoints use the offline v2 taxonomy.  The wrapper keeps both
specialist backbones, embeds ImageNet normalization and projects probabilities
to the deployed v1 contract.  Unsupported generic ``truck`` evidence is routed
to ``unknown``; fine colors are only merged into their explicit v1 parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))


BODY_V2 = [
    "sedan", "suv", "mpv", "van", "pickup", "truck", "bus",
    "light_truck", "heavy_truck", "other", "unknown",
]
COLOR_V2 = [
    "black", "white", "gray", "silver", "red", "blue", "green",
    "yellow", "brown", "other", "unknown",
]
BODY_V1 = [
    "sedan", "suv", "mpv", "van", "pickup", "bus", "light_truck",
    "heavy_truck", "other", "unknown",
]
COLOR_V1 = [
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown",
]

# Validation-only class thresholds selected by Stage168 and Stage159 R8.
BODY_THRESHOLDS = [0.657, 0.831, 0.882, 0.566, 0.922, 0.612, 0.835, 0.781, 0.923, 0.0]
COLOR_THRESHOLDS = [0.751, 0.519, 0.746, 0.626, 0.702, 0.590, 0.844, 0.500, 0.765, 0.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projection(source: list[str], target: list[str], mapping: dict[str, str]) -> torch.Tensor:
    matrix = torch.zeros(len(source), len(target), dtype=torch.float32)
    for source_index, label in enumerate(source):
        target_label = mapping.get(label, label)
        matrix[source_index, target.index(target_label)] = 1.0
    return matrix


class BestComponentsV1Runtime(nn.Module):
    def __init__(self, body_model: nn.Module, color_model: nn.Module, mean, std) -> None:
        super().__init__()
        self.body_model = body_model
        self.color_model = color_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            "body_projection",
            projection(BODY_V2, BODY_V1, {"truck": "unknown"}),
        )
        self.register_buffer(
            "color_projection",
            projection(
                COLOR_V2,
                COLOR_V1,
                {"gray": "silver_gray", "silver": "silver_gray",
                 "yellow": "yellow_orange", "brown": "brown_beige"},
            ),
        )
        self.register_buffer("body_thresholds", torch.tensor(BODY_THRESHOLDS))
        self.register_buffer("color_thresholds", torch.tensor(COLOR_THRESHOLDS))

    @staticmethod
    def gate(probabilities: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        winner = probabilities.argmax(dim=1)
        confidence = probabilities.gather(1, winner.unsqueeze(1)).squeeze(1)
        required = thresholds.gather(0, winner)
        unknown_index = probabilities.shape[1] - 1
        accepted = (winner == unknown_index) | (confidence >= required)
        unknown = torch.zeros_like(probabilities)
        unknown[:, unknown_index] = 1.0
        selected = torch.where(accepted.unsqueeze(1), probabilities, unknown)
        selected = selected.clamp_min(1.0e-7)
        selected = selected / selected.sum(dim=1, keepdim=True)
        return selected.log()

    def forward(self, images: torch.Tensor):
        normalized = (images - self.mean) / self.std
        body_logits, _ = self.body_model(normalized)
        _, color_logits = self.color_model(normalized)
        body_probabilities = torch.softmax(body_logits, dim=1) @ self.body_projection
        color_probabilities = torch.softmax(color_logits, dim=1) @ self.color_projection
        return (
            self.gate(body_probabilities, self.body_thresholds),
            self.gate(color_probabilities, self.color_thresholds),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--body-sha256", required=True)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--color-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise RuntimeError("refusing to overwrite candidate export evidence")
    if sha256(args.body_checkpoint).lower() != args.body_sha256.lower():
        raise RuntimeError("body checkpoint SHA256 mismatch")
    if sha256(args.color_checkpoint).lower() != args.color_sha256.lower():
        raise RuntimeError("color checkpoint SHA256 mismatch")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("body_types") != BODY_V1 or labels.get("colors") != COLOR_V1:
        raise RuntimeError("deployed v1 label contract mismatch")

    from src.multitask_mobilenet_v3 import model_from_checkpoint

    body_checkpoint = torch.load(args.body_checkpoint, map_location="cpu", weights_only=False)
    color_checkpoint = torch.load(args.color_checkpoint, map_location="cpu", weights_only=False)
    for checkpoint, role in ((body_checkpoint, "body"), (color_checkpoint, "color")):
        if checkpoint.get("body_types") != BODY_V2 or checkpoint.get("colors") != COLOR_V2:
            raise RuntimeError(f"{role} source taxonomy mismatch")
        if int(checkpoint.get("input_size", 0)) != 256 or checkpoint.get("resize_mode") != "stretch":
            raise RuntimeError(f"{role} geometry mismatch")

    body_model = model_from_checkpoint(body_checkpoint, pretrained=False)
    color_model = model_from_checkpoint(color_checkpoint, pretrained=False)
    body_model.load_state_dict(body_checkpoint["model_state"], strict=True)
    color_model.load_state_dict(color_checkpoint["model_state"], strict=True)
    mean = body_checkpoint["normalization"]["mean"]
    std = body_checkpoint["normalization"]["std"]
    if color_checkpoint["normalization"] != body_checkpoint["normalization"]:
        raise RuntimeError("specialist normalization mismatch")
    model = BestComponentsV1Runtime(body_model.eval(), color_model.eval(), mean, std).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        expected_body, expected_color = model(dummy.repeat(2, 1, 1, 1))
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["body_type", "color"],
        dynamic_axes={"images": {0: "batch"}, "body_type": {0: "batch"}, "color": {0: "batch"}},
        dynamo=False,
    )

    import numpy as np
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual_body, actual_color = session.run(None, {"images": np.zeros((2, 3, 256, 256), dtype=np.float32)})
    body_delta = float(np.max(np.abs(actual_body - expected_body.numpy())))
    color_delta = float(np.max(np.abs(actual_color - expected_color.numpy())))
    if actual_body.shape != (2, len(BODY_V1)) or actual_color.shape != (2, len(COLOR_V1)):
        raise RuntimeError("ONNX output dimensions do not match v1 runtime contract")
    if body_delta > 1.0e-4 or color_delta > 1.0e-4:
        raise RuntimeError("ONNX/PyTorch parity failed")

    report = {
        "schema_version": "best-components-v1-runtime-export-v1",
        "artifact_id": "vehicle-attr-best-components-256-r1",
        "body_source": {"path": str(args.body_checkpoint.resolve()), "sha256": sha256(args.body_checkpoint)},
        "color_source": {"path": str(args.color_checkpoint.resolve()), "sha256": sha256(args.color_checkpoint)},
        "input": {"width": 256, "height": 256, "channels": 3, "resize_mode": "stretch"},
        "outputs": {"body_type": BODY_V1, "color": COLOR_V1},
        "taxonomy_projection": {
            "generic_truck": "unknown",
            "gray_and_silver": "silver_gray",
            "yellow": "yellow_orange",
            "brown": "brown_beige",
            "fabricated_labels": False,
        },
        "thresholds_embedded": {"body_type": dict(zip(BODY_V1, BODY_THRESHOLDS)), "color": dict(zip(COLOR_V1, COLOR_THRESHOLDS))},
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256(args.output),
        "onnx_checker": "pass",
        "onnxruntime_cpu_smoke": "pass",
        "onnx_pytorch_max_abs_delta": {"body_type": body_delta, "color": color_delta},
        "policy": {"frozen_video_used": False, "production_files_overwritten": False},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
