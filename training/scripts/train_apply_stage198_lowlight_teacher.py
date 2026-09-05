#!/usr/bin/env python3
"""Train a grouped high-precision low-light appearance teacher and audit Stage177 unknown rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def group_key(row: dict[str, str]) -> str:
    return str(row.get("stage177_scene_group_key") or row.get("stage177_group_key") or row.get("image_path") or "")


def split_groups(rows: list[dict[str, str]], validation_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    clean: list[tuple[str, int, str, list[dict[str, str]]]] = []
    conflicts = 0
    for key, members in grouped.items():
        labels = {int(row["scene_target"]) for row in members}
        sources = {row["source_dataset"] for row in members}
        if len(labels) != 1 or len(sources) != 1:
            conflicts += len(members)
            continue
        clean.append((next(iter(sources)), next(iter(labels)), key, members))

    buckets: dict[tuple[str, int], list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for source, label, key, members in clean:
        buckets[(source, label)].append((key, members))
    train, validation = [], []
    for bucket_key, groups in sorted(buckets.items()):
        groups.sort(key=lambda item: stable_int(f"stage198|{bucket_key}|{item[0]}"))
        validation_groups = max(1, round(len(groups) * validation_fraction)) if len(groups) >= 2 else 0
        for index, (_, members) in enumerate(groups):
            (validation if index < validation_groups else train).extend(members)
    return train, validation, conflicts


def cap_training_negatives(rows: list[dict[str, str]], ratio: int, floor: int) -> list[dict[str, str]]:
    by_source_label: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source_label[(row["source_dataset"], int(row["scene_target"]))].append(row)
    output = []
    sources = sorted({source for source, _ in by_source_label})
    for source in sources:
        positives = by_source_label[(source, 1)]
        negatives = by_source_label[(source, 0)]
        output.extend(positives)
        limit = min(len(negatives), max(floor, ratio * max(1, len(positives))))
        negatives.sort(key=lambda row: stable_int(f"negative|{row['image_path']}"))
        output.extend(negatives[:limit])
    output.sort(key=lambda row: stable_int(f"selected|{row['image_path']}"))
    return output


def select_precision_threshold(probabilities: list[float], targets: list[int], minimum_precision: float, minimum_predictions: int) -> tuple[float | None, dict[str, float | int]]:
    order = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    positives = sum(targets)
    true_positive = 0
    best: tuple[float, dict[str, float | int]] | None = None
    for rank, index in enumerate(order, 1):
        true_positive += int(targets[index] == 1)
        next_probability = probabilities[order[rank]] if rank < len(order) else -1.0
        if probabilities[index] == next_probability:
            continue
        precision = true_positive / rank
        recall = true_positive / positives if positives else 0.0
        if rank >= minimum_predictions and precision >= minimum_precision:
            metrics = {"predicted_positive": rank, "true_positive": true_positive, "precision": precision, "recall": recall}
            if best is None or recall > float(best[1]["recall"]):
                best = (probabilities[index], metrics)
    return best if best is not None else (None, {"predicted_positive": 0, "true_positive": 0, "precision": 0.0, "recall": 0.0})


class SceneDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Any, flip: bool = False, with_target: bool = True) -> None:
        self.rows = rows
        self.transform = transform
        self.flip = flip
        self.with_target = with_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["absolute_path"]) as opened:
            image = opened.convert("RGB")
        if self.flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = self.transform(image)
        target = float(row["scene_target"]) if self.with_target else -1.0
        return tensor, target, index


def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    probabilities = [0.0] * len(loader.dataset)
    targets = [0] * len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for images, batch_targets, indexes in loader:
            logits = model(images.to(device, non_blocking=True)).flatten()
            batch_probabilities = logits.sigmoid().cpu().tolist()
            for index, probability, target in zip(indexes.tolist(), batch_probabilities, batch_targets.tolist()):
                probabilities[index] = float(probability)
                targets[index] = int(target)
    return probabilities, targets


def classification_metrics(probabilities: list[float], targets: list[int], threshold: float) -> dict[str, float | int]:
    predicted = [probability >= threshold for probability in probabilities]
    tp = sum(p and t == 1 for p, t in zip(predicted, targets))
    fp = sum(p and t == 0 for p, t in zip(predicted, targets))
    fn = sum((not p) and t == 1 for p, t in zip(predicted, targets))
    tn = sum((not p) and t == 0 for p, t in zip(predicted, targets))
    return {
        "rows": len(targets), "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "predicted_positive": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-precision", type=float, default=0.98)
    parser.add_argument("--minimum-validation-predictions", type=int, default=100)
    parser.add_argument("--minimum-group-size", type=int, default=3)
    parser.add_argument("--minimum-group-positive-fraction", type=float, default=0.8)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--negative-floor-per-source", type=int, default=500)
    parser.add_argument("--seed", type=int, default=198)
    args = parser.parse_args()

    for value in (args.manifest, args.allowed_root, args.output_root):
        if any(marker in str(value).lower() for marker in FROZEN_MARKERS):
            raise ValueError("frozen-video path is forbidden")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    actual_manifest_sha = sha256(args.manifest)
    if actual_manifest_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_manifest_sha}")
    allowed_root = args.allowed_root.resolve()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    labeled, unknown = [], []
    split_counts: Counter[str] = Counter()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_counts[str(row.get("split") or "")] += 1
            if row.get("split") != "train" or not truthy(row.get("stage177_dedup_eligible")) or not truthy(row.get("stage177_effective_representative")):
                continue
            searchable = " ".join(str(row.get(field) or "") for field in ("image_path", "source_manifest", "video_id", "source_frame_id")).lower()
            if any(marker in searchable for marker in FROZEN_MARKERS):
                raise RuntimeError("frozen marker found in eligible train row")
            path = Path(str(row.get("image_path") or ""))
            absolute = path.resolve() if path.is_absolute() else (args.manifest.parent / path).resolve()
            absolute.relative_to(allowed_root)
            prepared = dict(row)
            prepared["absolute_path"] = str(absolute)
            origin = str(row.get("stage177_scene_origin") or "")
            label = str(row.get("stage177_scene_label") or "")
            if origin in {"existing_explicit_night", "existing_explicit_low_light"} and label in {"night", "low_light"}:
                prepared["scene_target"] = "1"
                labeled.append(prepared)
            elif origin == "existing_explicit_daylight" and label == "daylight":
                prepared["scene_target"] = "0"
                labeled.append(prepared)
            elif label == "unknown":
                unknown.append(prepared)
    if not labeled or not unknown:
        raise RuntimeError("missing labeled or unknown Stage177 rows")

    train_rows, validation_rows, conflicting_group_rows = split_groups(labeled, args.validation_fraction)
    train_rows = cap_training_negatives(train_rows, args.negative_ratio, args.negative_floor_per_source)
    if len({row["scene_target"] for row in train_rows}) != 2 or len({row["scene_target"] for row in validation_rows}) != 2:
        raise RuntimeError("group split lost a class")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.78, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), antialias=True),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class_counts = Counter(int(row["scene_target"]) for row in train_rows)
    sample_weights = [1.0 / class_counts[int(row["scene_target"])] for row in train_rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(SceneDataset(train_rows, train_transform), batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    validation_loader = DataLoader(SceneDataset(validation_rows, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    args.output_root.mkdir(parents=True)
    epochs_log = []
    best_score = -1.0
    checkpoint_path = args.output_root / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True)).flatten()
            loss = criterion(logits, targets.to(device, non_blocking=True).float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
        probabilities, targets = infer(model, validation_loader, device)
        metrics = classification_metrics(probabilities, targets, 0.5)
        balanced_accuracy = 0.5 * (float(metrics["recall"]) + float(metrics["specificity"]))
        epoch_record = {"epoch": epoch, "train_loss": total_loss / len(train_rows), "validation_at_0_5": metrics, "balanced_accuracy": balanced_accuracy}
        epochs_log.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            torch.save({
                "schema_version": "stage198-lowlight-teacher-v1", "model_state": model.state_dict(),
                "input_size": args.input_size, "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
                "seed": args.seed, "epoch": epoch, "balanced_accuracy": balanced_accuracy,
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    validation_probabilities, validation_targets = infer(model, validation_loader, device)
    threshold, threshold_metrics = select_precision_threshold(
        validation_probabilities, validation_targets, args.minimum_precision, args.minimum_validation_predictions,
    )
    threshold_value = float(threshold) if threshold is not None else 1.0
    overall_validation = classification_metrics(validation_probabilities, validation_targets, threshold_value)
    per_source_validation = {}
    for source in sorted({row["source_dataset"] for row in validation_rows}):
        indexes = [index for index, row in enumerate(validation_rows) if row["source_dataset"] == source]
        per_source_validation[source] = classification_metrics(
            [validation_probabilities[index] for index in indexes],
            [validation_targets[index] for index in indexes],
            threshold_value,
        )
    supported_sources = [metrics for metrics in per_source_validation.values() if int(metrics["predicted_positive"]) >= 10]
    per_source_precision_floor = max(0.95, args.minimum_precision - 0.02)
    per_source_gate = len(supported_sources) >= 2 and all(float(metrics["precision"]) >= per_source_precision_floor for metrics in supported_sources)
    validation_gate = threshold is not None and per_source_gate

    unknown_original_loader = DataLoader(SceneDataset(unknown, eval_transform, flip=False, with_target=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    unknown_flip_loader = DataLoader(SceneDataset(unknown, eval_transform, flip=True, with_target=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    original_probabilities, _ = infer(model, unknown_original_loader, device)
    flip_probabilities, _ = infer(model, unknown_flip_loader, device)
    minimum_probabilities = [min(left, right) for left, right in zip(original_probabilities, flip_probabilities)]

    grouped_unknown: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(unknown):
        grouped_unknown[group_key(row)].append(index)
    accepted_indexes: set[int] = set()
    group_diagnostics: Counter[str] = Counter()
    if validation_gate:
        for indexes in grouped_unknown.values():
            positives = [index for index in indexes if minimum_probabilities[index] >= threshold_value]
            fraction = len(positives) / len(indexes)
            if len(indexes) < args.minimum_group_size:
                group_diagnostics["group_too_small"] += len(indexes)
            elif fraction < args.minimum_group_positive_fraction:
                group_diagnostics["group_consensus_failed"] += len(indexes)
            else:
                accepted_indexes.update(positives)
                group_diagnostics["accepted"] += len(positives)
                group_diagnostics["group_positive_but_frame_below_threshold"] += len(indexes) - len(positives)

    overlay_path = args.output_root / "stage198-unknown-lowlight-overlay.csv"
    overlay_fields = [
        "image_path", "source_dataset", "camera_id", "video_id", "track_group", "source_frame_id",
        "stage177_group_key", "stage177_scene_group_key", "stage177_scene_label", "stage177_scene_origin",
        "stage198_original_probability", "stage198_hflip_probability", "stage198_min_probability",
        "stage198_scene_label", "stage198_scene_origin", "stage198_scene_confidence", "stage198_train_eligible",
    ]
    source_accepts: Counter[str] = Counter()
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overlay_fields)
        writer.writeheader()
        for index, row in enumerate(unknown):
            accepted = index in accepted_indexes
            source_accepts[row["source_dataset"]] += int(accepted)
            writer.writerow({
                **{field: row.get(field, "") for field in overlay_fields},
                "stage198_original_probability": f"{original_probabilities[index]:.8f}",
                "stage198_hflip_probability": f"{flip_probabilities[index]:.8f}",
                "stage198_min_probability": f"{minimum_probabilities[index]:.8f}",
                "stage198_scene_label": "low_light" if accepted else "unknown",
                "stage198_scene_origin": "binary_teacher_group_consensus_proxy" if accepted else "fail_closed_unknown",
                "stage198_scene_confidence": f"{minimum_probabilities[index]:.8f}" if accepted else "0",
                "stage198_train_eligible": str(accepted).lower(),
            })

    checkpoint["validation_threshold"] = threshold
    checkpoint["validation_metrics"] = overall_validation
    checkpoint["manifest_sha256"] = actual_manifest_sha
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema_version": "stage198-lowlight-teacher-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_high_precision_proxy_audit" if validation_gate else "fail_closed_validation_precision_gate",
        "input": {"manifest": str(args.manifest.resolve()), "manifest_sha256": actual_manifest_sha, "split_counts": dict(split_counts)},
        "training": {
            "architecture": "mobilenet_v3_small", "pretrained_weights": "IMAGENET1K_V1", "input_size": args.input_size,
            "epochs": args.epochs, "seed": args.seed, "train_rows": len(train_rows), "train_class_counts": dict(class_counts),
            "validation_rows": len(validation_rows), "conflicting_group_rows_excluded": conflicting_group_rows,
            "epochs_log": epochs_log,
        },
        "calibration": {
            "minimum_precision": args.minimum_precision, "minimum_validation_predictions": args.minimum_validation_predictions,
            "selected_threshold": threshold, "selection_metrics": threshold_metrics,
            "overall_validation": overall_validation, "per_source_validation": per_source_validation,
            "per_source_precision_floor": per_source_precision_floor,
            "supported_sources_required": 2, "supported_sources_observed": len(supported_sources),
            "per_source_gate": per_source_gate,
        },
        "unknown_audit": {
            "unknown_effective_rows": len(unknown), "accepted_lowlight_proxy_rows": len(accepted_indexes),
            "accepted_by_source": dict(sorted(source_accepts.items())), "group_diagnostics": dict(sorted(group_diagnostics.items())),
            "minimum_group_size": args.minimum_group_size, "minimum_group_positive_fraction": args.minimum_group_positive_fraction,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "overlay": str(overlay_path), "overlay_sha256": sha256(overlay_path),
        },
        "policy": {
            "accepted_label_semantics": "low_light_proxy_only_never_confirmed_natural_night",
            "attribute_labels_modified": False, "source_manifest_modified": False,
            "validation_or_test_pixels_opened": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage198-lowlight-teacher-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if validation_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
