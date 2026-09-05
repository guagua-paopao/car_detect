#!/usr/bin/env python3
"""Create an isolated taxonomy-v2 initialization checkpoint from a v1 model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json  # noqa: E402
from src.multitask_mobilenet_v3 import (  # noqa: E402
    MultiTaskVehicleAttributes,
    architecture_from_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_layer_keys(state: dict, prefix: str, class_count: int) -> tuple[str, str]:
    candidates = [
        key
        for key, value in state.items()
        if key.startswith(prefix)
        and key.endswith(".weight")
        and getattr(value, "ndim", 0) == 2
        and int(value.shape[0]) == class_count
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {prefix} output weight with {class_count} rows, got {candidates}"
        )
    weight = candidates[0]
    bias = weight.removesuffix(".weight") + ".bias"
    if bias not in state or int(state[bias].shape[0]) != class_count:
        raise RuntimeError(f"matching output bias is missing for {weight}")
    return weight, bias


def source_recipe(target: str, old_labels: list[str], head: str) -> tuple[str, ...]:
    if target in old_labels:
        return (target,)
    if head == "color":
        aliases = {
            "gray": ("silver_gray",),
            "silver": ("silver_gray",),
            "yellow": ("yellow_orange",),
            "brown": ("brown_beige",),
        }
        return tuple(label for label in aliases.get(target, ()) if label in old_labels)
    if head == "body" and target == "truck":
        return tuple(label for label in ("light_truck", "heavy_truck") if label in old_labels)
    return ()


def copy_output_rows(
    source_state: dict,
    target_state: dict,
    prefix: str,
    old_labels: list[str],
    new_labels: list[str],
    head: str,
) -> dict[str, list[str]]:
    import torch

    source_weight, source_bias = output_layer_keys(source_state, prefix, len(old_labels))
    target_weight, target_bias = output_layer_keys(target_state, prefix, len(new_labels))
    audit: dict[str, list[str]] = {}
    with torch.no_grad():
        for target_index, target_label in enumerate(new_labels):
            recipe = source_recipe(target_label, old_labels, head)
            if not recipe:
                audit[target_label] = []
                continue
            source_indices = [old_labels.index(label) for label in recipe]
            target_state[target_weight][target_index].copy_(
                source_state[source_weight][source_indices].mean(dim=0)
            )
            target_state[target_bias][target_index].copy_(
                source_state[source_bias][source_indices].mean(dim=0)
            )
            audit[target_label] = list(recipe)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    labels = load_json(args.labels)
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 offline labels are required")
    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    old_body = [str(label) for label in checkpoint["body_types"]]
    old_colors = [str(label) for label in checkpoint["colors"]]
    new_body = [str(label) for label in labels["body_types"]]
    new_colors = [str(label) for label in labels["colors"]]
    architecture = architecture_from_checkpoint(checkpoint)
    model = MultiTaskVehicleAttributes(
        len(new_body), len(new_colors), architecture=architecture, pretrained=False
    )
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    copied_exact_tensors = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
            target_state[key] = source_value.clone()
            copied_exact_tensors.append(key)

    body_audit = copy_output_rows(
        source_state, target_state, "body_type_head", old_body, new_body, "body"
    )
    color_audit = copy_output_rows(
        source_state, target_state, "color_head", old_colors, new_colors, "color"
    )
    model.load_state_dict(target_state, strict=True)
    migrated = {
        "schema_version": "attribute-taxonomy-v2-init-v1",
        "epoch": -1,
        "best_score": -1.0,
        "gate_best_key": [-1.0, -1.0, -1.0],
        "epochs_without_improvement": 0,
        "model_state": model.state_dict(),
        "architecture": f"{architecture}_multitask",
        "input_size": int(checkpoint.get("input_size", 224)),
        "resize_mode": checkpoint.get("resize_mode", "stretch"),
        "body_types": new_body,
        "colors": new_colors,
        "labels_version": labels["labels_version"],
        "initialization": {
            "source_checkpoint": str(args.input_checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(args.input_checkpoint),
            "body_row_sources": body_audit,
            "color_row_sources": color_audit,
            "policy": "shared tensors copied exactly; new fine colors inherit the corresponding merged v1 row; generic truck starts from the mean of light/heavy truck and requires family-partial loss",
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output_checkpoint)
    report = {
        "schema_version": "attribute-taxonomy-v2-init-report-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_checkpoint": str(args.input_checkpoint.resolve()),
        "input_checkpoint_sha256": sha256(args.input_checkpoint),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256(args.output_checkpoint),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "architecture": architecture,
        "copied_exact_tensor_count": len(copied_exact_tensors),
        "body_row_sources": body_audit,
        "color_row_sources": color_audit,
        "deployment_status": "offline_candidate_only",
        "production_model_modified": False,
        "frozen_video_used": False,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
