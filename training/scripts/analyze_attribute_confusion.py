#!/usr/bin/env python3
"""Report per-class confusion for a trained vehicle attribute checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json, trainable_attribute_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "center_crop"),
        default="stretch",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(
    labels: list[str],
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    paths: list[str],
) -> dict:
    evaluated = [i for i, target in enumerate(targets) if target != -100]
    true_counts = Counter(labels[targets[i]] for i in evaluated)
    predicted_counts = Counter(labels[predictions[i]] for i in evaluated)
    confusion = Counter(
        (labels[targets[i]], labels[predictions[i]]) for i in evaluated
    )
    per_class = {}
    for label in labels:
        total = true_counts[label]
        correct = confusion[(label, label)]
        per_class[label] = {
            "support": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "predicted_as": {
                predicted: count
                for (actual, predicted), count in sorted(confusion.items())
                if actual == label
            },
        }
    mistakes = [
        {
            "path": paths[i],
            "actual": labels[targets[i]],
            "predicted": labels[predictions[i]],
            "confidence": round(confidences[i], 6),
        }
        for i in evaluated
        if targets[i] != predictions[i]
    ]
    return {
        "evaluated": len(evaluated),
        "accuracy": (
            sum(targets[i] == predictions[i] for i in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
        "true_counts": dict(sorted(true_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
        "mistakes": mistakes,
    }


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.attribute_dataset import VehicleAttributeDataset
    from src.multitask_mobilenet_v3 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        model_from_checkpoint,
    )

    labels_config = load_json(args.labels)
    body_types, colors = trainable_attribute_labels(labels_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["body_types"] != body_types or checkpoint["colors"] != colors:
        raise ValueError("checkpoint label order does not match local project config")
    input_size = int(checkpoint.get("input_size", 224))
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
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = VehicleAttributeDataset(
        args.manifest,
        split=args.split,
        body_types=body_types,
        colors=colors,
        transform=transform,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device(args.device)
    model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    body_targets: list[int] = []
    body_predictions: list[int] = []
    body_confidences: list[float] = []
    color_targets: list[int] = []
    color_predictions: list[int] = []
    color_confidences: list[float] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, body_target, color_target, batch_paths in loader:
            body_logits, color_logits = model(images.to(device))
            body_probabilities = body_logits.softmax(dim=1).cpu()
            color_probabilities = color_logits.softmax(dim=1).cpu()
            body_confidence, body_prediction = body_probabilities.max(dim=1)
            color_confidence, color_prediction = color_probabilities.max(dim=1)
            body_targets.extend(body_target.tolist())
            body_predictions.extend(body_prediction.tolist())
            body_confidences.extend(body_confidence.tolist())
            color_targets.extend(color_target.tolist())
            color_predictions.extend(color_prediction.tolist())
            color_confidences.extend(color_confidence.tolist())
            paths.extend(batch_paths)

    report = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "resize_mode": args.resize_mode,
        "samples": len(dataset),
        "body_type": summarize(
            body_types,
            body_targets,
            body_predictions,
            body_confidences,
            paths,
        ),
        "color": summarize(
            colors,
            color_targets,
            color_predictions,
            color_confidences,
            paths,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
