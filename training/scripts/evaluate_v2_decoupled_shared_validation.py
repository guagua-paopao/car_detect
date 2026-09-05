#!/usr/bin/env python3
"""Validation-only evaluation for a decoupled taxonomy-v2 body/color pair.

The real-CCTV VFG-7 truth uses the production shared-color taxonomy, so this
evaluator aggregates gray/silver safely and reports exact and truck-family
body metrics separately. It has no test or frozen-video input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))


class SquarePad:
    """Pad a PIL image to a centered square using the training fill value."""

    def __init__(self, fill: tuple[int, int, int] = (114, 114, 114)) -> None:
        self.fill = fill

    def __call__(self, image):
        from PIL import ImageOps

        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        border = (
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        )
        return ImageOps.expand(image, border=border, fill=self.fill)


SHARED_COLORS = (
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown",
)

PRODUCTION_V1_LABELS = {
    "labels_version": "vehicle-labels-v1",
    "body_types": [
        "sedan",
        "suv",
        "mpv",
        "van",
        "pickup",
        "bus",
        "light_truck",
        "heavy_truck",
        "other",
        "unknown",
    ],
    "colors": list(SHARED_COLORS),
}
TRUCK_FAMILY = {"truck", "light_truck", "heavy_truck"}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def is_complex_row(row: dict[str, str]) -> bool:
    lighting = str(row.get("lighting") or "").strip().lower()
    weather = str(row.get("weather") or "").strip().lower()
    occlusion = str(row.get("occlusion_level") or "").strip().lower()
    quality = str(row.get("crop_quality") or "").strip().lower()
    return any((
        truthy(row.get("night")),
        truthy(row.get("low_light")),
        truthy(row.get("blur")),
        truthy(row.get("occluded")),
        truthy(row.get("truncated")),
        str(row.get("vehicle_size") or "").strip().lower() == "small",
        lighting in {"night", "low_light", "backlight", "strong_backlight"},
        weather not in {"", "clear", "normal", "unknown"},
        occlusion not in {"", "none", "clear", "unknown"},
        quality in {"poor", "rejected"},
    ))


def aggregate_color_probabilities(labels: list[str], probabilities: Iterable[float]) -> dict[str, float]:
    source = dict(zip(labels, (float(value) for value in probabilities)))
    v2 = {"black", "white", "gray", "silver", "red", "blue", "green", "yellow", "brown", "other", "unknown"}
    v1 = set(SHARED_COLORS)
    if set(labels) == v1:
        return {label: source[label] for label in SHARED_COLORS}
    if set(labels) == v2:
        return {
            "black": source["black"],
            "white": source["white"],
            "silver_gray": source["gray"] + source["silver"],
            "red": source["red"],
            "blue": source["blue"],
            "green": source["green"],
            "yellow_orange": source["yellow"],
            "brown_beige": source["brown"],
            "other": source["other"],
            "unknown": source["unknown"],
        }
    raise ValueError(f"unexpected color labels: {labels}")


def top_label(scores: dict[str, float]) -> tuple[str, float]:
    label = max(scores, key=scores.get)
    return label, float(scores[label])


def body_correct(prediction: str, truth: str, *, family_aware: bool) -> bool:
    if prediction == truth:
        return True
    return family_aware and prediction in TRUCK_FAMILY and truth in TRUCK_FAMILY


def metric_at_threshold(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    threshold: float,
    *,
    family_aware: bool = False,
) -> dict[str, Any]:
    if not (len(predictions) == len(confidences) == len(truths)):
        raise ValueError("metric inputs have different lengths")
    selected = [
        index for index, (prediction, confidence) in enumerate(zip(predictions, confidences))
        if prediction != "unknown" and confidence >= threshold
    ]
    correct = sum(body_correct(predictions[index], truths[index], family_aware=family_aware) for index in selected)
    evaluated = len(truths)
    return {
        "threshold": threshold,
        "evaluated": evaluated,
        "selected": len(selected),
        "correct": correct,
        "precision": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / evaluated if evaluated else 0.0,
        "unknown_rate": 1.0 - len(selected) / evaluated if evaluated else 1.0,
    }


def per_class_metrics(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    if not (len(predictions) == len(confidences) == len(truths)):
        raise ValueError("per-class metric inputs have different lengths")
    emitted = [
        prediction if prediction != "unknown" and confidence >= threshold else "unknown"
        for prediction, confidence in zip(predictions, confidences)
    ]
    result: dict[str, dict[str, Any]] = {}
    for label in sorted(set(truths) | {value for value in emitted if value != "unknown"}):
        if label == "unknown":
            continue
        truth_indexes = [index for index, truth in enumerate(truths) if truth == label]
        predicted_indexes = [index for index, prediction in enumerate(emitted) if prediction == label]
        selected_truth = [index for index in truth_indexes if emitted[index] != "unknown"]
        true_positives = sum(emitted[index] == label for index in truth_indexes)
        support = len(truth_indexes)
        result[label] = {
            "support": support,
            "predicted": len(predicted_indexes),
            "true_positives": true_positives,
            "precision": true_positives / len(predicted_indexes) if predicted_indexes else 0.0,
            "recall": true_positives / support if support else 0.0,
            "coverage": len(selected_truth) / support if support else 0.0,
            "unknown_rate": 1.0 - len(selected_truth) / support if support else 1.0,
        }
    return result


def select_threshold(
    predictions: list[str],
    confidences: list[float],
    truths: list[str],
    candidates: Iterable[float],
    *,
    precision_gate: float,
    family_aware: bool = False,
) -> dict[str, Any]:
    trials = [
        metric_at_threshold(predictions, confidences, truths, float(threshold), family_aware=family_aware)
        for threshold in candidates
    ]
    passing = [trial for trial in trials if trial["selected"] > 0 and trial["precision"] >= precision_gate]
    if passing:
        chosen = max(passing, key=lambda item: (item["coverage"], item["precision"], -item["threshold"]))
        return {**chosen, "precision_gate_found": True, "trials": trials}
    chosen = max(trials, key=lambda item: (item["precision"], item["coverage"], -item["threshold"]))
    return {**chosen, "precision_gate_found": False, "trials": trials}


def quality_weight(row: dict[str, str]) -> float:
    value = 1.0
    if row.get("vehicle_size") == "small":
        value *= 0.70
    if truthy(row.get("night")) or truthy(row.get("low_light")) or row.get("lighting") in {"night", "low_light"}:
        value *= 0.80
    if truthy(row.get("blur")):
        value *= 0.75
    if truthy(row.get("occluded")) or truthy(row.get("truncated")):
        value *= 0.65
    if row.get("crop_quality") in {"poor", "rejected"}:
        value *= 0.50
    return max(0.10, value)


def fuse_observations(
    observations: list[tuple[str, float, float]],
    *,
    window: int = 5,
    minimum_share: float = 0.60,
    minimum_margin: float = 0.15,
) -> list[str]:
    history: deque[tuple[str, float, float]] = deque(maxlen=window)
    outputs: list[str] = []
    for observation in observations:
        history.append(observation)
        scores: Counter[str] = Counter()
        total = 0.0
        for label, confidence, quality in history:
            if label == "unknown":
                continue
            weight = max(0.0, confidence) * max(0.0, quality)
            scores[label] += weight
            total += weight
        if not scores or total <= 0:
            outputs.append("unknown")
            continue
        ranked = scores.most_common(2)
        winner, winner_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        share = winner_score / total
        margin = (winner_score - runner_up) / total
        outputs.append(winner if share >= minimum_share and margin >= minimum_margin else "unknown")
    return outputs


def stability(labels_by_track: dict[str, list[str]]) -> dict[str, Any]:
    emitted = 0
    transitions = 0
    switches = 0
    dominant_sum = 0
    for labels in labels_by_track.values():
        known = [label for label in labels if label != "unknown"]
        emitted += len(known)
        if known:
            dominant_sum += max(Counter(known).values())
        transitions += max(0, len(known) - 1)
        switches += sum(left != right for left, right in zip(known, known[1:]))
    return {
        "tracks": len(labels_by_track),
        "emitted": emitted,
        "label_switches": switches,
        "transition_stability": 1.0 - switches / transitions if transitions else 1.0,
        "dominant_label_stability": dominant_sum / emitted if emitted else 1.0,
    }


def load_model(checkpoint_path: Path, device: Any):
    import torch
    from src.multitask_mobilenet_v3 import model_from_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = model_from_checkpoint(checkpoint, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def validate_checkpoint_taxonomy(
    body_checkpoint: dict[str, Any],
    color_checkpoint: dict[str, Any],
    expected_labels: dict[str, Any] | None,
) -> None:
    expected = expected_labels if expected_labels is not None else PRODUCTION_V1_LABELS
    role = "candidate" if expected_labels is not None else "production baseline"
    if (
        body_checkpoint.get("labels_version") != expected.get("labels_version")
        or color_checkpoint.get("labels_version") != expected.get("labels_version")
        or body_checkpoint.get("body_types") != expected.get("body_types")
        or color_checkpoint.get("colors") != expected.get("colors")
    ):
        raise RuntimeError(f"{role} checkpoint taxonomy mismatch")
    if expected_labels is not None:
        if body_checkpoint.get("body_hierarchy") != "truck_family":
            raise RuntimeError("candidate body checkpoint lost truck-family hierarchy")
        subtype_threshold = float(body_checkpoint.get("truck_subtype_threshold") or 0.0)
        if not 0.0 < subtype_threshold <= 1.0:
            raise RuntimeError("candidate truck subtype threshold is invalid")


def validate_truck_specialist_checkpoint(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("truck specialist labels version mismatch")
    if checkpoint.get("body_types") != ["light_truck", "heavy_truck", "unknown"]:
        raise RuntimeError("truck specialist body taxonomy mismatch")
    if checkpoint.get("colors") != ["unknown"]:
        raise RuntimeError("truck specialist color taxonomy mismatch")


def route_body_with_specialist(
    full_logits: Any,
    full_labels: list[str],
    specialist_logits: Any,
    specialist_labels: list[str],
    subtype_threshold: float,
):
    """Route only a winning truck family through an exact subtype expert.

    An uncertain subtype becomes ``unknown`` for the exact-type gate.  The
    coarse family probability is never relabelled as light/heavy evidence.
    """
    import torch

    if full_logits.ndim != 2 or specialist_logits.ndim != 2:
        raise ValueError("router logits must have shape [batch, classes]")
    if full_logits.shape[0] != specialist_logits.shape[0]:
        raise ValueError("router batch sizes differ")
    if specialist_labels != ["light_truck", "heavy_truck", "unknown"]:
        raise ValueError("unexpected truck specialist labels")
    if not 0.0 < subtype_threshold <= 1.0:
        raise ValueError("specialist subtype threshold must be in (0, 1]")
    hierarchy = {label: full_labels.index(label) for label in (*TRUCK_FAMILY, "unknown")}
    family_indices = torch.tensor(
        [hierarchy["truck"], hierarchy["light_truck"], hierarchy["heavy_truck"]],
        device=full_logits.device,
        dtype=torch.long,
    )
    outside_mask = torch.ones(len(full_labels), device=full_logits.device, dtype=torch.bool)
    outside_mask[family_indices] = False
    full_probabilities = torch.softmax(full_logits, dim=1)
    predictions = full_probabilities.argmax(dim=1)
    confidences = full_probabilities.amax(dim=1)
    family_mass = full_probabilities.index_select(1, family_indices).sum(dim=1)
    outside_best = full_probabilities[:, outside_mask].amax(dim=1)
    choose_family = family_mass > outside_best

    specialist_probabilities = torch.softmax(specialist_logits, dim=1)
    best_subtype_probability, subtype_offset = specialist_probabilities[:, :2].max(dim=1)
    accept_subtype = choose_family & (best_subtype_probability >= subtype_threshold)
    reject_subtype = choose_family & ~accept_subtype
    subtype_full_indices = torch.tensor(
        [hierarchy["light_truck"], hierarchy["heavy_truck"]],
        device=full_logits.device,
        dtype=torch.long,
    )
    predictions = predictions.clone()
    confidences = confidences.clone()
    predictions[accept_subtype] = subtype_full_indices[subtype_offset[accept_subtype]]
    # A geometric joint confidence penalizes either an uncertain family or an
    # uncertain subtype while remaining calibratable by the validation gate.
    confidences[accept_subtype] = torch.sqrt(
        family_mass[accept_subtype] * best_subtype_probability[accept_subtype]
    )
    predictions[reject_subtype] = hierarchy["unknown"]
    confidences[reject_subtype] = family_mass[reject_subtype]
    return predictions, confidences


def resolve_rows(manifest: Path, safety_root: Path) -> list[dict[str, str]]:
    safety_root = safety_root.resolve()
    if not safety_root.is_dir():
        raise RuntimeError("datasets safety root is missing")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("validation manifest has no rows")
    for line, row in enumerate(rows, start=2):
        if row.get("split") != "validation":
            raise RuntimeError(f"validation-only manifest contains split={row.get('split')} at line {line}")
        searchable = " ".join(row.values()).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"frozen marker in validation row {line}")
        if row.get("review_status") != "approved":
            raise RuntimeError(f"unapproved validation row {line}")
        raw = Path(row.get("image_path", ""))
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"validation image escapes datasets safety root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["_resolved_image_path"] = str(resolved)
    return rows


def infer_pair(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    body_checkpoint_path: Path,
    color_checkpoint_path: Path,
    *,
    expected_labels: dict[str, Any] | None,
    body_specialist_checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from src.attribute_hierarchy import decode_truck_family, truck_hierarchy

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    body_model, body_checkpoint = load_model(body_checkpoint_path, device)
    color_model, color_checkpoint = load_model(color_checkpoint_path, device)
    specialist_model = None
    specialist_checkpoint = None
    if body_specialist_checkpoint_path is not None:
        specialist_model, specialist_checkpoint = load_model(
            body_specialist_checkpoint_path, device
        )
        validate_truck_specialist_checkpoint(specialist_checkpoint)
    validate_checkpoint_taxonomy(body_checkpoint, color_checkpoint, expected_labels)
    if int(body_checkpoint.get("input_size", 0)) != int(color_checkpoint.get("input_size", 0)):
        raise RuntimeError("decoupled checkpoints require the same input size")
    if body_checkpoint.get("resize_mode", "stretch") != color_checkpoint.get(
        "resize_mode", "stretch"
    ):
        raise RuntimeError("decoupled checkpoints require the same resize mode")
    input_size = int(body_checkpoint["input_size"])
    body_hierarchy = (
        truck_hierarchy(body_checkpoint["body_types"])
        if body_checkpoint.get("body_hierarchy") == "truck_family"
        else None
    )
    truck_subtype_threshold = float(body_checkpoint.get("truck_subtype_threshold") or 0.65)
    def geometry(checkpoint: dict[str, Any]):
        size = int(checkpoint["input_size"])
        mode = checkpoint.get("resize_mode", "stretch")
        if mode == "stretch":
            operations = [transforms.Resize((size, size), antialias=True)]
        elif mode == "center_crop":
            operations = [
                transforms.Resize(round(size * 232 / 224), antialias=True),
                transforms.CenterCrop(size),
            ]
        elif mode == "letterbox":
            operations = [SquarePad(), transforms.Resize((size, size), antialias=True)]
        else:
            raise RuntimeError(f"unsupported resize mode: {mode}")
        return transforms.Compose(
            [
                *operations,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    body_transform = geometry(body_checkpoint)
    specialist_transform = geometry(specialist_checkpoint) if specialist_checkpoint else None

    class Images(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            with Image.open(rows[index]["_resolved_image_path"]) as image:
                rgb = image.convert("RGB")
                body_image = body_transform(rgb)
                if specialist_transform is not None:
                    return body_image, specialist_transform(rgb), index
                return body_image, index

    loader = DataLoader(Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    outputs: list[dict[str, Any] | None] = [None] * len(rows)
    with torch.no_grad():
        for batch in loader:
            if specialist_transform is not None:
                images, specialist_images, indexes = batch
                specialist_images = specialist_images.to(device, non_blocking=True)
            else:
                images, indexes = batch
                specialist_images = None
            images = images.to(device, non_blocking=True)
            body_logits, _ = body_model(images)
            _, color_logits = color_model(images)
            if specialist_model is not None and specialist_checkpoint is not None:
                if specialist_images is None:
                    raise RuntimeError("specialist transform was not initialized")
                specialist_logits, _ = specialist_model(specialist_images)
                body_indices, body_confidences = route_body_with_specialist(
                    body_logits,
                    body_checkpoint["body_types"],
                    specialist_logits,
                    specialist_checkpoint["body_types"],
                    float(args.body_specialist_subtype_threshold),
                )
                body_indices = body_indices.cpu().tolist()
                body_confidences = body_confidences.cpu().tolist()
            elif body_hierarchy is not None:
                body_indices, body_confidences = decode_truck_family(
                    body_logits, body_hierarchy, truck_subtype_threshold
                )
                body_indices = body_indices.cpu().tolist()
                body_confidences = body_confidences.cpu().tolist()
            else:
                body_probabilities = torch.softmax(body_logits, dim=1)
                body_confidences, body_indices = body_probabilities.max(dim=1)
                body_indices = body_indices.cpu().tolist()
                body_confidences = body_confidences.cpu().tolist()
            color_probs = torch.softmax(color_logits, dim=1).cpu().tolist()
            for position, raw_index in enumerate(indexes.tolist()):
                body_label = body_checkpoint["body_types"][body_indices[position]]
                body_confidence = float(body_confidences[position])
                color_scores = aggregate_color_probabilities(color_checkpoint["colors"], color_probs[position])
                color_label, color_confidence = top_label(color_scores)
                outputs[raw_index] = {
                    "body_label": body_label,
                    "body_confidence": body_confidence,
                    "color_label": color_label,
                    "color_confidence": color_confidence,
                }
    if any(value is None for value in outputs):
        raise RuntimeError("inference did not cover every row")
    return [value for value in outputs if value is not None]


def stratified_metrics(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    threshold: float,
    *,
    family_aware: bool = False,
) -> dict[str, Any]:
    fields = (
        "lighting", "vehicle_size", "weather", "occlusion_level",
        "source_dataset", "source_video",
    )
    result: dict[str, Any] = {}
    for field in fields:
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            supervised = truthy(row.get("body_type_supervised" if head == "body" else "color_supervised"))
            if supervised:
                groups[row.get(field) or "unknown"].append(index)
        result[field] = {}
        for key, indexes in sorted(groups.items()):
            truths = [rows[index]["body_type" if head == "body" else "color"] for index in indexes]
            predictions = [outputs[index][f"{head}_label"] for index in indexes]
            confidences = [outputs[index][f"{head}_confidence"] for index in indexes]
            result[field][key] = metric_at_threshold(predictions, confidences, truths, threshold, family_aware=family_aware)
    return result


def fused_metrics(
    rows: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    head: str,
    threshold: float,
    *,
    family_aware: bool = False,
) -> dict[str, Any]:
    tracks: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if truthy(row.get("body_type_supervised" if head == "body" else "color_supervised")):
            tracks[row.get("track_key") or row.get("track_id") or f"row:{index}"].append(index)
    fused_by_index: dict[int, str] = {}
    labels_by_track: dict[str, list[str]] = {}
    track_truths: dict[str, str] = {}
    for track, indexes in tracks.items():
        indexes.sort(key=lambda index: int(float(rows[index].get("window_pos") or 0)))
        observations = []
        for index in indexes:
            label = outputs[index][f"{head}_label"]
            confidence = float(outputs[index][f"{head}_confidence"])
            if confidence < threshold:
                label = "unknown"
            observations.append((label, confidence, quality_weight(rows[index])))
        fused = fuse_observations(observations)
        labels_by_track[track] = fused
        fused_by_index.update(zip(indexes, fused))
        truth_field = "track_body_truth" if head == "body" else "track_color_truth"
        row_truth_field = "body_type" if head == "body" else "color"
        track_truths[track] = rows[indexes[0]].get(truth_field) or rows[indexes[0]][row_truth_field]
    selected_indexes = sorted(fused_by_index)
    predictions = [fused_by_index[index] for index in selected_indexes]
    confidences = [1.0 if prediction != "unknown" else 0.0 for prediction in predictions]
    truths = [rows[index]["body_type" if head == "body" else "color"] for index in selected_indexes]
    frame_metric = metric_at_threshold(predictions, confidences, truths, 0.5, family_aware=family_aware)
    ordered_tracks = sorted(labels_by_track)
    final_predictions = [labels_by_track[track][-1] if labels_by_track[track] else "unknown" for track in ordered_tracks]
    final_confidences = [1.0 if label != "unknown" else 0.0 for label in final_predictions]
    final_truths = [track_truths[track] for track in ordered_tracks]
    track_metric = metric_at_threshold(
        final_predictions, final_confidences, final_truths, 0.5, family_aware=family_aware
    )
    return {
        "frame": frame_metric,
        "track_final": track_metric,
        "stability": stability(labels_by_track),
        "window": 5,
        "minimum_share": 0.60,
        "minimum_margin": 0.15,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--body-specialist-checkpoint", type=Path)
    parser.add_argument("--expected-body-specialist-checkpoint-sha256")
    parser.add_argument("--body-specialist-subtype-threshold", type=float, default=0.80)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-color-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision-gate", type=float, default=0.93)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation evidence: {args.output}")
    if (args.body_specialist_checkpoint is None) != (
        args.expected_body_specialist_checkpoint_sha256 is None
    ):
        raise RuntimeError("body specialist checkpoint and expected SHA256 must be provided together")
    pinned = [
        (args.manifest, args.expected_manifest_sha256),
        (args.labels, args.expected_labels_sha256),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256),
        (args.color_checkpoint, args.expected_color_checkpoint_sha256),
        (args.baseline_checkpoint, args.expected_baseline_checkpoint_sha256),
    ]
    if args.body_specialist_checkpoint is not None:
        pinned.append((
            args.body_specialist_checkpoint,
            args.expected_body_specialist_checkpoint_sha256,
        ))
    for path, expected in pinned:
        if not path.is_file() or sha256(path).lower() != expected.lower():
            raise RuntimeError(f"immutable validation input mismatch: {path}")
    lowered = str(args.manifest).lower()
    if "test" in lowered or "60s" in lowered or "36-48" in lowered:
        raise RuntimeError("non-validation manifest path rejected")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("taxonomy-v2 labels are required")
    rows = resolve_rows(args.manifest, args.datasets_safety_root)
    outputs = infer_pair(
        args, rows, args.body_checkpoint, args.color_checkpoint, expected_labels=labels,
        body_specialist_checkpoint_path=args.body_specialist_checkpoint,
    )
    baseline_outputs = infer_pair(
        args, rows, args.baseline_checkpoint, args.baseline_checkpoint, expected_labels=None
    )
    thresholds = [value / 1000 for value in range(500, 991)]

    supervised_body = [index for index, row in enumerate(rows) if truthy(row.get("body_type_supervised"))]
    supervised_color = [index for index, row in enumerate(rows) if truthy(row.get("color_supervised"))]
    complex_body = [index for index in supervised_body if is_complex_row(rows[index])]
    complex_color = [index for index in supervised_color if is_complex_row(rows[index])]
    if not complex_body or not complex_color:
        raise RuntimeError("complex validation subset is empty")
    body_predictions = [outputs[index]["body_label"] for index in supervised_body]
    body_confidences = [outputs[index]["body_confidence"] for index in supervised_body]
    body_truths = [rows[index]["body_type"] for index in supervised_body]
    color_predictions = [outputs[index]["color_label"] for index in supervised_color]
    color_confidences = [outputs[index]["color_confidence"] for index in supervised_color]
    color_truths = [rows[index]["color"] for index in supervised_color]
    baseline_body_predictions = [baseline_outputs[index]["body_label"] for index in supervised_body]
    baseline_body_confidences = [baseline_outputs[index]["body_confidence"] for index in supervised_body]
    baseline_color_predictions = [baseline_outputs[index]["color_label"] for index in supervised_color]
    baseline_color_confidences = [baseline_outputs[index]["color_confidence"] for index in supervised_color]
    body_selection = select_threshold(body_predictions, body_confidences, body_truths, thresholds, precision_gate=args.precision_gate)
    body_family_selection = select_threshold(body_predictions, body_confidences, body_truths, thresholds, precision_gate=args.precision_gate, family_aware=True)
    color_selection = select_threshold(color_predictions, color_confidences, color_truths, thresholds, precision_gate=args.precision_gate)
    baseline_body_selection = select_threshold(
        baseline_body_predictions, baseline_body_confidences, body_truths, thresholds,
        precision_gate=args.precision_gate,
    )
    baseline_color_selection = select_threshold(
        baseline_color_predictions, baseline_color_confidences, color_truths, thresholds,
        precision_gate=args.precision_gate,
    )

    body_threshold = float(body_selection["threshold"])
    body_family_threshold = float(body_family_selection["threshold"])
    color_threshold = float(color_selection["threshold"])
    baseline_body_threshold = float(baseline_body_selection["threshold"])
    baseline_color_threshold = float(baseline_color_selection["threshold"])
    candidate_static = {
        "body_exact": metric_at_threshold(body_predictions, body_confidences, body_truths, body_threshold),
        "body_family": metric_at_threshold(
            body_predictions, body_confidences, body_truths, body_family_threshold, family_aware=True
        ),
        "color_shared": metric_at_threshold(color_predictions, color_confidences, color_truths, color_threshold),
    }
    baseline_static = {
        "body_exact": metric_at_threshold(
            baseline_body_predictions, baseline_body_confidences, body_truths, baseline_body_threshold
        ),
        "color_shared": metric_at_threshold(
            baseline_color_predictions, baseline_color_confidences, color_truths, baseline_color_threshold
        ),
    }
    candidate_complex = {
        "body_exact": metric_at_threshold(
            [outputs[index]["body_label"] for index in complex_body],
            [outputs[index]["body_confidence"] for index in complex_body],
            [rows[index]["body_type"] for index in complex_body],
            body_threshold,
        ),
        "color_shared": metric_at_threshold(
            [outputs[index]["color_label"] for index in complex_color],
            [outputs[index]["color_confidence"] for index in complex_color],
            [rows[index]["color"] for index in complex_color],
            color_threshold,
        ),
    }
    baseline_complex = {
        "body_exact": metric_at_threshold(
            [baseline_outputs[index]["body_label"] for index in complex_body],
            [baseline_outputs[index]["body_confidence"] for index in complex_body],
            [rows[index]["body_type"] for index in complex_body],
            baseline_body_threshold,
        ),
        "color_shared": metric_at_threshold(
            [baseline_outputs[index]["color_label"] for index in complex_color],
            [baseline_outputs[index]["color_confidence"] for index in complex_color],
            [rows[index]["color"] for index in complex_color],
            baseline_color_threshold,
        ),
    }
    candidate_tracks = {
        "body_exact": fused_metrics(rows, outputs, "body", body_threshold),
        "body_family": fused_metrics(rows, outputs, "body", body_family_threshold, family_aware=True),
        "color_shared": fused_metrics(rows, outputs, "color", color_threshold),
    }
    baseline_tracks = {
        "body_exact": fused_metrics(rows, baseline_outputs, "body", baseline_body_threshold),
        "color_shared": fused_metrics(rows, baseline_outputs, "color", baseline_color_threshold),
    }
    body_coverage_gain = candidate_complex["body_exact"]["coverage"] - baseline_complex["body_exact"]["coverage"]
    baseline_complex_color_unknown = baseline_complex["color_shared"]["unknown_rate"]
    candidate_complex_color_unknown = candidate_complex["color_shared"]["unknown_rate"]
    color_unknown_reduction = (
        (baseline_complex_color_unknown - candidate_complex_color_unknown) / baseline_complex_color_unknown
        if baseline_complex_color_unknown > 0 else 0.0
    )
    baseline_track_color_unknown = baseline_tracks["color_shared"]["track_final"]["unknown_rate"]
    candidate_track_color_unknown = candidate_tracks["color_shared"]["track_final"]["unknown_rate"]
    track_color_unknown_reduction = (
        (baseline_track_color_unknown - candidate_track_color_unknown) / baseline_track_color_unknown
        if baseline_track_color_unknown > 0 else 0.0
    )
    shared_gates = {
        "body_static_precision": candidate_static["body_exact"]["precision"] >= 0.93,
        "body_static_coverage": candidate_static["body_exact"]["coverage"] >= 0.45,
        "body_complex_coverage_gain": body_coverage_gain >= 0.15,
        "color_static_precision": candidate_static["color_shared"]["precision"] >= 0.93,
        "color_static_coverage": candidate_static["color_shared"]["coverage"] >= 0.25,
        "color_complex_unknown_reduction": color_unknown_reduction >= 0.20,
        "body_track_precision": candidate_tracks["body_exact"]["track_final"]["precision"] >= 0.93,
        "body_track_coverage": candidate_tracks["body_exact"]["track_final"]["coverage"] >= 0.45,
        "color_track_precision": candidate_tracks["color_shared"]["track_final"]["precision"] >= 0.93,
        "color_track_coverage": candidate_tracks["color_shared"]["track_final"]["coverage"] >= 0.25,
        "body_track_stability": candidate_tracks["body_exact"]["stability"]["transition_stability"] >= 0.95,
        "color_track_stability": candidate_tracks["color_shared"]["stability"]["transition_stability"] >= 0.95,
    }
    shared_gates_pass = all(shared_gates.values())
    report = {
        "schema_version": "v2-decoupled-shared-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256(args.labels),
            "body_checkpoint": str(args.body_checkpoint.resolve()),
            "body_checkpoint_sha256": sha256(args.body_checkpoint),
            "body_specialist_checkpoint": (
                str(args.body_specialist_checkpoint.resolve())
                if args.body_specialist_checkpoint else None
            ),
            "body_specialist_checkpoint_sha256": (
                sha256(args.body_specialist_checkpoint)
                if args.body_specialist_checkpoint else None
            ),
            "body_specialist_subtype_threshold": (
                args.body_specialist_subtype_threshold
                if args.body_specialist_checkpoint else None
            ),
            "color_checkpoint": str(args.color_checkpoint.resolve()),
            "color_checkpoint_sha256": sha256(args.color_checkpoint),
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        },
        "rows": {
            "validation": len(rows),
            "body_supervised": len(supervised_body),
            "color_supervised": len(supervised_color),
            "complex_body_supervised": len(complex_body),
            "complex_color_supervised": len(complex_color),
        },
        "taxonomy_policy": {
            "color": "gray+silver probabilities are summed to silver_gray; yellow maps to yellow_orange and brown to brown_beige without treating other as orange/beige",
            "body_exact": "generic truck is not credited as an exact light/heavy subtype",
            "body_family_diagnostic": "truck/light_truck/heavy_truck are also reported as a separate family-aware diagnostic",
            "body_inference": (
                "candidate truck-family probability routes to a separately trained exact light/heavy specialist; uncertain subtype becomes unknown"
                if args.body_specialist_checkpoint
                else "candidate truck-family logits use the checkpoint-pinned hierarchical decoder and subtype threshold; production baseline remains flat taxonomy-v1"
            ),
        },
        "threshold_selection": {
            "source": "VFG-7 validation only",
            "body_exact": body_selection,
            "body_family": body_family_selection,
            "color_shared": color_selection,
            "baseline_body": baseline_body_selection,
            "baseline_color": baseline_color_selection,
        },
        "static": {"candidate": candidate_static, "production_baseline": baseline_static},
        "complex_static": {
            "definition": "night/low-light/backlight, adverse weather, small target, blur, occlusion, truncation or poor crop quality",
            "candidate": candidate_complex,
            "production_baseline": baseline_complex,
        },
        "per_class": {
            "candidate": {
                "body_exact": per_class_metrics(body_predictions, body_confidences, body_truths, body_threshold),
                "color_shared": per_class_metrics(color_predictions, color_confidences, color_truths, color_threshold),
            },
            "production_baseline": {
                "body_exact": per_class_metrics(
                    baseline_body_predictions, baseline_body_confidences, body_truths, baseline_body_threshold
                ),
                "color_shared": per_class_metrics(
                    baseline_color_predictions, baseline_color_confidences, color_truths, baseline_color_threshold
                ),
            },
        },
        "stratified": {
            "body_exact": stratified_metrics(rows, outputs, "body", body_threshold),
            "body_family": stratified_metrics(rows, outputs, "body", body_family_threshold, family_aware=True),
            "color_shared": stratified_metrics(rows, outputs, "color", color_threshold),
        },
        "track_fusion": {
            "candidate": candidate_tracks,
            "production_baseline": baseline_tracks,
        },
        "comparison": {
            "body_complex_static_coverage_gain": body_coverage_gain,
            "color_complex_static_unknown_relative_reduction": color_unknown_reduction,
            "color_track_unknown_relative_reduction_diagnostic": track_color_unknown_reduction,
        },
        "shared_validation_gates": {
            "gates": shared_gates,
            "all_pass": shared_gates_pass,
            "decision": (
                "shared_cctv_validation_pass_pending_exact_fine_color_and_generic_truck_evidence"
                if shared_gates_pass
                else "candidate_rejected_shared_cctv_validation_gate_failure"
            ),
            "not_proven_by_this_report": [
                "exact gray versus silver CCTV precision/coverage",
                "exact yellow versus orange or brown versus beige CCTV precision/coverage",
                "independent generic-truck exact truth",
                "independent test, backend, ONNX, TensorRT or frozen-video gates",
            ],
        },
        "policy": {
            "split": "validation",
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": report["status"],
        "body": report["static"]["candidate"]["body_exact"],
        "color": report["static"]["candidate"]["color_shared"],
        "track_stability": {
            "body": report["track_fusion"]["candidate"]["body_exact"]["stability"],
            "color": report["track_fusion"]["candidate"]["color_shared"]["stability"],
        },
        "shared_gates_pass": report["shared_validation_gates"]["all_pass"],
        "test_accessed": False,
        "frozen_video_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
