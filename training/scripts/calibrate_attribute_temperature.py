#!/usr/bin/env python3
"""Calibrate attribute logits while preserving the fixed runtime thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--body-threshold", type=float, default=0.75)
    parser.add_argument("--color-threshold", type=float, default=0.70)
    parser.add_argument("--target-precision", type=float, default=0.93)
    parser.add_argument(
        "--selection-precision-margin",
        type=float,
        default=0.0,
        help=(
            "Extra validation-only precision margin used when selecting a "
            "temperature. Release acceptance is still evaluated against "
            "--target-precision."
        ),
    )
    parser.add_argument("--body-selection-precision-margin", type=float, default=None)
    parser.add_argument("--color-selection-precision-margin", type=float, default=None)
    parser.add_argument("--minimum-selected", type=int, default=5)
    parser.add_argument("--minimum-coverage", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "center_crop"),
        default="stretch",
    )
    return parser.parse_args()


def collect_logits(model, loader, device) -> dict[str, Any]:
    import torch

    values: dict[str, list[Any]] = {
        "body_logits": [],
        "body_targets": [],
        "color_logits": [],
        "color_targets": [],
    }
    model.eval()
    with torch.no_grad():
        for images, body_target, color_target, _ in loader:
            body_logits, color_logits = model(images.to(device))
            body_mask = body_target != -100
            color_mask = color_target != -100
            values["body_logits"].append(body_logits[body_mask].cpu())
            values["body_targets"].append(body_target[body_mask].cpu())
            values["color_logits"].append(color_logits[color_mask].cpu())
            values["color_targets"].append(color_target[color_mask].cpu())
    return {
        key: torch.cat(items)
        for key, items in values.items()
    }


def metrics_for_temperature(
    logits,
    targets,
    *,
    temperature: float,
    threshold: float,
    unknown_index: int,
) -> dict[str, Any]:
    probabilities = (logits / temperature).softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    selected = (prediction != unknown_index) & (confidence >= threshold)
    selected_count = int(selected.sum())
    precision = (
        float((prediction[selected] == targets[selected]).float().mean())
        if selected_count
        else 0.0
    )
    return {
        "temperature": temperature,
        "selected": selected_count,
        "coverage": selected_count / len(targets) if len(targets) else 0.0,
        "precision": precision,
        "accuracy": float((prediction == targets).float().mean()),
    }


def calibrate_head(
    validation_logits,
    validation_targets,
    test_logits,
    test_targets,
    *,
    threshold: float,
    unknown_index: int,
    target_precision: float,
    selection_precision_margin: float,
    minimum_selected: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    selection_target_precision = target_precision + selection_precision_margin
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be between 0 and 1")
    if not 0.0 <= selection_target_precision <= 1.0:
        raise ValueError(
            "target_precision + selection_precision_margin must be between 0 and 1"
        )
    candidates = [round(0.25 + index * 0.025, 6) for index in range(191)]
    table = [
        metrics_for_temperature(
            validation_logits,
            validation_targets,
            temperature=temperature,
            threshold=threshold,
            unknown_index=unknown_index,
        )
        for temperature in candidates
    ]
    passing = [
        row
        for row in table
        if row["selected"] >= minimum_selected
        and row["precision"] >= selection_target_precision
        and row["coverage"] >= minimum_coverage
    ]
    if passing:
        selected = max(
            passing,
            key=lambda row: (row["selected"], row["precision"], row["temperature"]),
        )
        meets_target = True
    else:
        selected = next(row for row in table if row["temperature"] == 1.0)
        meets_target = False
    test = metrics_for_temperature(
        test_logits,
        test_targets,
        temperature=float(selected["temperature"]),
        threshold=threshold,
        unknown_index=unknown_index,
    )
    return {
        "threshold": threshold,
        "target_precision": target_precision,
        "selection_precision_margin": selection_precision_margin,
        "selection_target_precision": selection_target_precision,
        "minimum_selected": minimum_selected,
        "minimum_coverage": minimum_coverage,
        "temperature": selected["temperature"],
        "validation": selected,
        "test": test,
        "validation_meets_target": meets_target,
        "test_meets_target": (
            test["selected"] >= minimum_selected
            and test["precision"] >= target_precision
            and test["coverage"] >= minimum_coverage
        ),
        "table": table,
    }


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("install training/requirements-cloud.txt first") from exc

    from src.attribute_dataset import VehicleAttributeDataset
    from src.multitask_mobilenet_v3 import model_from_checkpoint

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    labels = load_json(args.labels)
    body_types = list(checkpoint["body_types"])
    colors = list(checkpoint["colors"])
    if body_types != list(labels["body_types"]) or colors != list(labels["colors"]):
        raise RuntimeError("checkpoint labels do not match the local label contract")
    input_size = int(checkpoint["input_size"])
    if args.resize_mode == "stretch":
        geometry_transforms = [
            transforms.Resize((input_size, input_size), antialias=True)
        ]
    else:
        resize_size = round(input_size * 232 / 224)
        geometry_transforms = [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(input_size),
        ]
    transform = transforms.Compose(
        [
            *geometry_transforms,
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["normalization"]["mean"],
                checkpoint["normalization"]["std"],
            ),
        ]
    )
    datasets = {
        split: VehicleAttributeDataset(
            args.manifest,
            split=split,
            body_types=body_types,
            colors=colors,
            transform=transform,
            training=False,
        )
        for split in ("validation", "test")
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    collected = {
        split: collect_logits(
            model,
            DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4),
            device,
        )
        for split, dataset in datasets.items()
    }
    calibration = {
        "schema_version": "1.0",
        "method": "per_head_temperature_scaling",
        "checkpoint": str(args.checkpoint.resolve()),
        "resize_mode": args.resize_mode,
        "body_type": calibrate_head(
            collected["validation"]["body_logits"],
            collected["validation"]["body_targets"],
            collected["test"]["body_logits"],
            collected["test"]["body_targets"],
            threshold=args.body_threshold,
            unknown_index=body_types.index("unknown"),
            target_precision=args.target_precision,
            selection_precision_margin=(args.body_selection_precision_margin if args.body_selection_precision_margin is not None else args.selection_precision_margin),
            minimum_selected=args.minimum_selected,
            minimum_coverage=0.45,
        ),
        "color": calibrate_head(
            collected["validation"]["color_logits"],
            collected["validation"]["color_targets"],
            collected["test"]["color_logits"],
            collected["test"]["color_targets"],
            threshold=args.color_threshold,
            unknown_index=colors.index("unknown"),
            target_precision=args.target_precision,
            selection_precision_margin=(args.color_selection_precision_margin if args.color_selection_precision_margin is not None else args.selection_precision_margin),
            minimum_selected=args.minimum_selected,
            minimum_coverage=0.25,
        ),
    }
    calibration["release_target_met"] = all(
        calibration[field]["validation_meets_target"]
        and calibration[field]["test_meets_target"]
        for field in ("body_type", "color")
    )
    write_json(args.output.resolve(), calibration)
    print(json.dumps(calibration, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
