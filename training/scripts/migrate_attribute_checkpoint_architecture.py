#!/usr/bin/env python3
"""Create an offline cross-head MobileNetV3 initialization checkpoint.

The attribute-tuned backbone is copied bit-for-bit.  Architecture-specific
projection heads remain deterministic fresh parameters; no class row is
invented when the source and target head geometry differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import (  # noqa: E402
    MultiTaskVehicleAttributes,
    architecture_from_checkpoint,
)


MOBILENET_V3_FAMILY = {
    "mobilenet_v3_large",
    "mobilenet_v3_large_dual",
    "mobilenet_v3_large_foreground_dual",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_checkpoint(checkpoint: dict, target_architecture: str, seed: int) -> tuple[dict, dict]:
    import torch

    source_architecture = architecture_from_checkpoint(checkpoint)
    if source_architecture not in MOBILENET_V3_FAMILY:
        raise RuntimeError(f"source architecture is not MobileNetV3-compatible: {source_architecture}")
    if target_architecture not in MOBILENET_V3_FAMILY:
        raise RuntimeError(f"target architecture is not MobileNetV3-compatible: {target_architecture}")
    body_types = [str(value) for value in checkpoint["body_types"]]
    colors = [str(value) for value in checkpoint["colors"]]
    torch.manual_seed(seed)
    model = MultiTaskVehicleAttributes(
        len(body_types), len(colors), architecture=target_architecture, pretrained=False
    )
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    target_backbone_keys = [key for key in target_state if key.startswith("backbone.")]
    copied_backbone: list[str] = []
    for key in target_backbone_keys:
        source_value = source_state.get(key)
        if source_value is None or tuple(source_value.shape) != tuple(target_state[key].shape):
            raise RuntimeError(f"backbone tensor is missing or incompatible: {key}")
        target_state[key] = source_value.detach().clone()
        copied_backbone.append(key)
    copied_heads: list[str] = []
    for key in target_state:
        if key.startswith("backbone."):
            continue
        source_value = source_state.get(key)
        if source_value is not None and tuple(source_value.shape) == tuple(target_state[key].shape):
            target_state[key] = source_value.detach().clone()
            copied_heads.append(key)
    model.load_state_dict(target_state, strict=True)
    migrated = {
        "schema_version": "attribute-architecture-init-v1",
        "epoch": -1,
        "best_score": -1.0,
        "gate_best_key": [-1.0, -1.0, -1.0],
        "epochs_without_improvement": 0,
        "model_state": model.state_dict(),
        "architecture": f"{target_architecture}_multitask",
        "input_size": int(checkpoint.get("input_size", 224)),
        "resize_mode": checkpoint.get("resize_mode", "stretch"),
        "body_types": body_types,
        "colors": colors,
        "labels_version": checkpoint.get("labels_version"),
        "initialization": {
            "source_architecture": source_architecture,
            "target_architecture": target_architecture,
            "seed": seed,
            "copied_backbone_tensor_count": len(copied_backbone),
            "copied_head_tensors": copied_heads,
            "policy": "copy every shape-compatible MobileNetV3 backbone tensor exactly; retain deterministic fresh target-head parameters when geometry differs",
        },
    }
    audit = {
        "source_architecture": source_architecture,
        "target_architecture": target_architecture,
        "seed": seed,
        "target_backbone_tensor_count": len(target_backbone_keys),
        "copied_backbone_tensor_count": len(copied_backbone),
        "copied_head_tensors": copied_heads,
        "fresh_head_tensors": [
            key for key in target_state if not key.startswith("backbone.") and key not in copied_heads
        ],
    }
    return migrated, audit


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--target-architecture", choices=sorted(MOBILENET_V3_FAMILY), required=True)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite architecture-migration evidence")
    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    migrated, audit = migrate_checkpoint(checkpoint, args.target_architecture, args.seed)
    migrated["initialization"].update({
        "source_checkpoint": str(args.input_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.input_checkpoint),
    })
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output_checkpoint)
    report = {
        "schema_version": "attribute-architecture-init-report-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_checkpoint": str(args.input_checkpoint.resolve()),
        "input_checkpoint_sha256": sha256(args.input_checkpoint),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256(args.output_checkpoint),
        **audit,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
