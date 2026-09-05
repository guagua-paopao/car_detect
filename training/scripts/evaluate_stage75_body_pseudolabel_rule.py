#!/usr/bin/env python3
"""Calibrate a conservative multi-teacher body pseudo-label rule on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=/absolute/path.pt")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return name, path


def assert_validation_asset(path: Path, label: str) -> Path:
    resolved = path.resolve()
    lowered = str(resolved).lower()
    if "vcas_rtsp_demo_60s" in lowered or "36-48" in lowered:
        raise RuntimeError(f"{label} references the frozen video: {resolved}")
    return resolved


def resolve_image_path(raw_value: str, dataset_root: Path, manifest_parent: Path) -> Path:
    raw = Path(raw_value)
    candidates = [raw] if raw.is_absolute() else [dataset_root / raw, manifest_parent / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        resolved.relative_to(dataset_root)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"image path not found under the validation root: {raw_value}")


def wilson_lower(correct: int, selected: int, z: float = 1.96) -> float:
    if selected <= 0:
        return 0.0
    probability = correct / selected
    denominator = 1.0 + z * z / selected
    centre = probability + z * z / (2.0 * selected)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / selected
        + z * z / (4.0 * selected * selected)
    )
    return max(0.0, (centre - radius) / denominator)


def aggregate_views(
    views: list[tuple[str, float]], minimum_views: int, confidence_floor: float
) -> tuple[str, float] | None:
    if len(views) < minimum_views:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for label, confidence in views:
        grouped[label].append(float(confidence))
    label, confidences = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    if len(confidences) < minimum_views:
        return None
    confidence = min(confidences)
    if confidence < confidence_floor:
        return None
    return label, confidence


def aggregate_teachers(
    teacher_views: dict[str, list[tuple[str, float]]],
    minimum_views: int,
    confidence_floor: float,
) -> tuple[str, float] | None:
    decisions = [
        aggregate_views(views, minimum_views, confidence_floor)
        for views in teacher_views.values()
    ]
    if not decisions or any(decision is None for decision in decisions):
        return None
    known = [decision for decision in decisions if decision is not None]
    labels = {decision[0] for decision in known}
    if len(labels) != 1:
        return None
    return known[0][0], min(decision[1] for decision in known)


def evaluate_configuration(
    rows: list[dict[str, str]],
    teacher_predictions: list[dict[str, list[tuple[str, float]]]],
    *,
    minimum_views: int,
    confidence_floor: float,
    known_labels: list[str],
) -> dict[str, Any]:
    selected = 0
    correct = 0
    per_truth: dict[str, Counter[str]] = {label: Counter() for label in known_labels}
    per_predicted: dict[str, Counter[str]] = {label: Counter() for label in known_labels}
    for row, teacher_views in zip(rows, teacher_predictions):
        truth = row["body_type"]
        per_truth.setdefault(truth, Counter())["evaluated"] += 1
        decision = aggregate_teachers(teacher_views, minimum_views, confidence_floor)
        if decision is None or decision[0] not in known_labels:
            continue
        predicted, _ = decision
        selected += 1
        per_truth[truth]["selected"] += 1
        per_predicted[predicted]["selected"] += 1
        if predicted == truth:
            correct += 1
            per_truth[truth]["correct"] += 1
            per_predicted[predicted]["correct"] += 1

    truth_metrics: dict[str, dict[str, float | int]] = {}
    predicted_metrics: dict[str, dict[str, float | int]] = {}
    classes_meeting_gate = 0
    for label in known_labels:
        truth_counter = per_truth.get(label, Counter())
        truth_evaluated = int(truth_counter["evaluated"])
        truth_selected = int(truth_counter["selected"])
        truth_correct = int(truth_counter["correct"])
        truth_metrics[label] = {
            "evaluated": truth_evaluated,
            "selected": truth_selected,
            "correct": truth_correct,
            "selected_accuracy": truth_correct / truth_selected if truth_selected else 0.0,
            "coverage": truth_selected / truth_evaluated if truth_evaluated else 0.0,
        }
        predicted_counter = per_predicted.get(label, Counter())
        predicted_selected = int(predicted_counter["selected"])
        predicted_correct = int(predicted_counter["correct"])
        precision = predicted_correct / predicted_selected if predicted_selected else 0.0
        if predicted_selected >= 10 and precision >= 0.90:
            classes_meeting_gate += 1
        predicted_metrics[label] = {
            "selected": predicted_selected,
            "correct": predicted_correct,
            "precision": precision,
        }
    precision = correct / selected if selected else 0.0
    coverage = selected / len(rows) if rows else 0.0
    return {
        "parameters": {
            "minimum_views": minimum_views,
            "confidence_floor": confidence_floor,
        },
        "evaluated": len(rows),
        "selected": selected,
        "correct": correct,
        "precision": precision,
        "precision_wilson_lower_95": wilson_lower(correct, selected),
        "coverage": coverage,
        "unknown_rate": 1.0 - coverage,
        "classes_with_at_least_10_and_precision_0_90": classes_meeting_gate,
        "per_truth_class": truth_metrics,
        "per_predicted_class": predicted_metrics,
    }


def passes_gate(result: dict[str, Any]) -> bool:
    return (
        int(result["selected"]) >= 150
        and float(result["precision"]) >= 0.96
        and float(result["precision_wilson_lower_95"]) >= 0.93
        and int(result["classes_with_at_least_10_and_precision_0_90"]) >= 5
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 3:
        raise ValueError("at least three independently trained checkpoints are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    manifest = assert_validation_asset(args.manifest, "manifest")
    dataset_root = assert_validation_asset(args.dataset_root, "dataset root")
    labels_path = assert_validation_asset(args.labels, "labels")
    labels_sha = sha256(labels_path)
    if labels_sha.lower() != args.expected_labels_sha256.lower():
        raise RuntimeError("label contract SHA256 mismatch")
    body_labels = json.loads(labels_path.read_text(encoding="utf-8"))["body_types"]
    known_labels = [label for label in body_labels if label not in {"unknown", "other"}]

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if row.get("split") == "validation"
            and truthy(row.get("body_type_supervised"))
            and row.get("body_type") in known_labels
        ]
    if not rows:
        raise RuntimeError("no supervised validation body rows")
    if any("test" in str(row.get("image_path", "")).lower() for row in rows):
        raise RuntimeError("validation rows include a test path")

    training_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(training_root))
    from src.multitask_mobilenet_v3 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        model_from_checkpoint,
    )
    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    paths: list[Path] = []
    for row in rows:
        paths.append(resolve_image_path(row["image_path"], dataset_root, manifest.parent))
    views = ("original", "hflip", "dim")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    teacher_predictions: list[dict[str, list[tuple[str, float]]]] = [
        defaultdict(list) for _ in rows
    ]
    checkpoint_evidence: dict[str, dict[str, Any]] = {}

    class ValidationDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int):
            with Image.open(paths[index]) as opened:
                image = opened.convert("RGB")
            if self.view == "hflip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = ImageEnhance.Contrast(
                    ImageEnhance.Brightness(image).enhance(0.80)
                ).enhance(1.08)
            return self.transform(image), index

    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if list(checkpoint.get("body_types", [])) != list(body_labels):
            raise RuntimeError(f"{name} body contract mismatch")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        for view in views:
            loader = DataLoader(
                ValidationDataset(transform, view), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, indexes in loader:
                    logits, _ = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for index, prediction, score in zip(
                        indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()
                    ):
                        teacher_predictions[index][name].append(
                            (body_labels[int(prediction)], float(score))
                        )
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidates: list[dict[str, Any]] = []
    for minimum_views in (3, 2):
        for confidence_floor in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            result = evaluate_configuration(
                rows,
                teacher_predictions,
                minimum_views=minimum_views,
                confidence_floor=confidence_floor,
                known_labels=known_labels,
            )
            result["passes_gate"] = passes_gate(result)
            candidates.append(result)
    passing = [candidate for candidate in candidates if candidate["passes_gate"]]
    selected = max(
        passing,
        key=lambda item: (float(item["coverage"]), float(item["precision"]), int(item["selected"])),
        default=None,
    )
    report = {
        "schema_version": "stage75-body-pseudolabel-rule-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_validation_rule_available" if selected else "fail_closed_no_rule_passed",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "labels": str(labels_path),
        "labels_sha256": labels_sha,
        "rows": len(rows),
        "truth_counts": dict(sorted(Counter(row["body_type"] for row in rows).items())),
        "checkpoints": checkpoint_evidence,
        "gate": {
            "minimum_selected": 150,
            "minimum_precision": 0.96,
            "minimum_precision_wilson_lower_95": 0.93,
            "minimum_classes_with_10_and_precision_0_90": 5,
        },
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda item: (bool(item["passes_gate"]), float(item["coverage"]), float(item["precision"])),
            reverse=True,
        ),
        "policy": {
            "usage": "validation_threshold_selection_only",
            "truth_used_as_teacher_input": False,
            "all_teachers_exact_agreement": True,
            "views": list(views),
            "unknown_and_other_never_pseudolabeled": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": (
            "freeze the selected rule before train-only UA-DETRAC track mining"
            if selected else
            "do not create body pseudo labels; acquire stronger independent teachers or labeled CCTV truth"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rows": len(rows), "selected": selected,
        "output": str(args.output), "output_sha256": sha256(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
