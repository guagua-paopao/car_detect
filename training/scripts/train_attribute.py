#!/usr/bin/env python3
"""Train the MobileNetV3-Large vehicle body-type/color multitask model."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import (  # noqa: E402
    load_json,
    sha256_file,
    trainable_attribute_labels,
    write_json,
)
from src.attribute_hierarchy import (  # noqa: E402
    car_family_indices,
    coarse_color_group_indices,
    coarse_truck_family_indices,
    decode_truck_family,
    partial_coarse_family_loss,
    partial_label_group_loss,
    partial_truck_family_loss,
    truck_family_match,
    truck_hierarchy,
)


class RandomJPEGCompression:
    """Round-trip a PIL image through bounded lossy JPEG compression."""

    def __init__(self, probability: float, minimum_quality: int, maximum_quality: int) -> None:
        self.probability = probability
        self.minimum_quality = minimum_quality
        self.maximum_quality = maximum_quality

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        from PIL import Image

        quality = random.randint(self.minimum_quality, self.maximum_quality)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class RandomColorTemperature:
    """Apply a small opposing red/blue gain without changing the class hue."""

    def __init__(self, probability: float, maximum_shift: float) -> None:
        self.probability = probability
        self.maximum_shift = maximum_shift

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        import numpy as np
        from PIL import Image

        shift = random.uniform(-self.maximum_shift, self.maximum_shift)
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        gains = np.asarray([1.0 + shift, 1.0, 1.0 - shift], dtype=np.float32)
        array *= gains.reshape(1, 1, 3)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


class RandomShadowHighlight:
    """Add a soft achromatic shadow or reflection band over the vehicle crop."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        import numpy as np
        from PIL import Image

        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = array.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx = xx / max(1, width - 1) - 0.5
        yy = yy / max(1, height - 1) - 0.5
        angle = random.uniform(0.0, math.pi)
        coordinate = math.cos(angle) * xx + math.sin(angle) * yy
        center = random.uniform(-0.30, 0.30)
        spread = random.uniform(0.12, 0.30)
        mask = np.exp(-0.5 * ((coordinate - center) / spread) ** 2)[..., None]
        if random.random() < 0.5:
            strength = random.uniform(0.18, 0.42)
            array *= 1.0 - strength * mask
        else:
            strength = random.uniform(0.08, 0.22)
            array += (255.0 - array) * strength * mask
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


class SquarePad:
    """Pad a PIL image to a centered square without changing its geometry."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument(
        "--architecture",
        choices=(
            "mobilenet_v3_large",
            "mobilenet_v3_large_dual",
            "mobilenet_v3_large_foreground_dual",
            "efficientnet_v2_s",
            "resnet50",
            "convnext_tiny",
        ),
        default="mobilenet_v3_large",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "center_crop", "letterbox"),
        default="stretch",
        help=(
            "Image geometry used by training/evaluation. 'stretch' matches the "
            "current TensorRT attribute adapter; 'letterbox' preserves vehicle aspect ratio."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--color-loss-weight", type=float, default=0.7)
    parser.add_argument("--body-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--color-focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse_sqrt"),
        default="inverse_sqrt",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze the ImageNet backbone for the first N epochs.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--type-threshold", type=float, default=0.75)
    parser.add_argument("--color-threshold", type=float, default=0.70)
    parser.add_argument("--gate-type-precision", type=float, default=0.93)
    parser.add_argument("--gate-type-coverage", type=float, default=0.45)
    parser.add_argument("--gate-color-precision", type=float, default=0.93)
    parser.add_argument("--gate-color-coverage", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights only, while resetting optimizer, scheduler, and epoch.",
    )
    parser.add_argument("--teacher-checkpoint", type=Path, help="Optional frozen multitask teacher for both-head logit distillation.")
    parser.add_argument("--body-teacher-checkpoint", type=Path, help="Optional frozen teacher used only for body-head distillation.")
    parser.add_argument("--color-teacher-checkpoint", type=Path, help="Optional frozen teacher used only for color-head distillation.")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-body-weight", type=float, default=1.0)
    parser.add_argument("--distill-color-weight", type=float, default=1.0)
    parser.add_argument(
        "--unlabeled-manifest",
        type=Path,
        help=(
            "Optional train-only manifest whose rows are explicitly unknown and "
            "unsupervised. It is used only for augmentation-consistency loss."
        ),
    )
    parser.add_argument(
        "--unlabeled-root",
        type=Path,
        help="Image-path root for --unlabeled-manifest (defaults to its parent).",
    )
    parser.add_argument("--unlabeled-batch-size", type=int, default=64)
    parser.add_argument("--unlabeled-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--unlabeled-consistency-mode",
        choices=("logit", "feature"),
        default="logit",
        help=(
            "logit matches EMA class distributions; feature matches normalized "
            "pre-head embeddings and cannot reinforce an unknown class decision"
        ),
    )
    parser.add_argument("--unlabeled-body-weight", type=float, default=1.0)
    parser.add_argument("--unlabeled-color-weight", type=float, default=1.0)
    parser.add_argument("--unlabeled-consistency-temperature", type=float, default=1.0)
    parser.add_argument("--unlabeled-ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--unlabeled-night-sample-weight",
        type=float,
        default=1.0,
        help="Relative sampler weight for rows marked night=true in the unlabeled stream.",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=("baseline", "hard_scene", "color_scene"),
        default="baseline",
        help="Use stronger illumination/blur augmentation for hard scenes.",
    )
    parser.add_argument(
        "--selection-head",
        choices=("joint", "body", "color"),
        default="joint",
        help="Metric head used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--body-hierarchy",
        choices=("none", "truck_family"),
        default="none",
        help=(
            "truck_family treats official generic truck truth as a partial family label "
            "and decodes a generic fallback when the fine subtype is uncertain"
        ),
    )
    parser.add_argument(
        "--truck-subtype-threshold",
        type=float,
        default=0.65,
        help="Minimum conditional light/heavy probability before emitting a fine truck subtype.",
    )
    parser.add_argument(
        "--coarse-car-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional partial-label weight for rows marked coarse_body_family=car. "
            "Their loss supervises summed sedan/SUV/MPV/other probability only."
        ),
    )
    parser.add_argument(
        "--coarse-truck-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional partial-label weight for rows marked coarse_body_family=truck. "
            "Their loss supervises summed generic/light/heavy truck probability when available."
        ),
    )
    parser.add_argument(
        "--coarse-color-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional partial-label weight for rows marked with a legacy merged "
            "coarse_color_group. silver_gray supervises summed gray/silver "
            "probability without fabricating an exact class."
        ),
    )
    parser.add_argument(
        "--night-sample-weight",
        type=float,
        default=1.0,
        help="Relative supervised sampler weight for real rows marked night=true.",
    )
    parser.add_argument(
        "--occlusion-sample-weight",
        type=float,
        default=1.0,
        help="Relative supervised sampler weight for occluded or truncated rows.",
    )
    parser.add_argument(
        "--hard-sample-weight",
        type=float,
        default=0.0,
        help="Extra sampling weight per hard_score unit; 0 disables hard mining.",
    )
    parser.add_argument(
        "--small-sample-weight",
        type=float,
        default=0.0,
        help="Extra multiplicative sampling weight for rows marked small_target or vehicle_size=small.",
    )
    parser.add_argument(
        "--color-sample-weight",
        type=float,
        default=0.0,
        help="Extra multiplicative sampling weight for rows with supervised non-unknown color labels.",
    )
    parser.add_argument(
        "--sampler-color-class-weighting",
        choices=("none", "inverse_sqrt"),
        default="none",
        help="Optional train-only color-class multiplier applied by the weighted sampler.",
    )
    parser.add_argument(
        "--sampler-color-class-weight-cap",
        type=float,
        default=4.0,
        help="Maximum per-row multiplier from --sampler-color-class-weighting.",
    )
    parser.add_argument(
        "--pseudo-label-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for rows marked pseudo_label=true; validation/test are never pseudo-weighted.",
    )
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--dataset-version", default="dataset-pilot-v1")
    parser.add_argument("--code-revision", default=os.environ.get("VCAS_CODE_REVISION", "unknown"))
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not load or evaluate the test split during iterative candidate development.",
    )
    return parser.parse_args()


def macro_f1(predictions: list[int], targets: list[int], class_count: int) -> float:
    scores: list[float] = []
    for class_id in range(class_count):
        true_positive = sum(
            pred == class_id and target == class_id
            for pred, target in zip(predictions, targets)
        )
        false_positive = sum(
            pred == class_id and target != class_id
            for pred, target in zip(predictions, targets)
        )
        false_negative = sum(
            pred != class_id and target == class_id
            for pred, target in zip(predictions, targets)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            scores.append(2 * true_positive / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(
    model,
    loader,
    device,
    body_count: int,
    color_count: int,
    body_unknown_index: int,
    color_unknown_index: int,
    type_threshold: float,
    color_threshold: float,
    body_hierarchy=None,
    truck_subtype_threshold: float = 0.65,
) -> dict[str, float]:
    import torch

    model.eval()
    body_predictions: list[int] = []
    body_targets: list[int] = []
    color_predictions: list[int] = []
    color_targets: list[int] = []
    body_confidences: list[float] = []
    color_confidences: list[float] = []
    with torch.no_grad():
        for images, body_target, color_target, _ in loader:
            images = images.to(device)
            body_logits, color_logits = model(images)
            if body_hierarchy is None:
                body_pred = body_logits.argmax(dim=1)
                body_confidence = body_logits.softmax(dim=1).amax(dim=1)
            else:
                body_pred, body_confidence = decode_truck_family(
                    body_logits, body_hierarchy, truck_subtype_threshold
                )
            body_pred = body_pred.cpu()
            color_pred = color_logits.argmax(dim=1).cpu()
            body_confidence = body_confidence.cpu()
            color_confidence = color_logits.softmax(dim=1).amax(dim=1).cpu()
            for pred, target, confidence in zip(
                body_pred.tolist(),
                body_target.tolist(),
                body_confidence.tolist(),
            ):
                if target != -100:
                    body_predictions.append(pred)
                    body_targets.append(target)
                    body_confidences.append(confidence)
            for pred, target, confidence in zip(
                color_pred.tolist(),
                color_target.tolist(),
                color_confidence.tolist(),
            ):
                if target != -100:
                    color_predictions.append(pred)
                    color_targets.append(target)
                    color_confidences.append(confidence)
    body_accuracy = (
        sum(a == b for a, b in zip(body_predictions, body_targets))
        / len(body_targets)
        if body_targets
        else 0.0
    )
    color_accuracy = (
        sum(a == b for a, b in zip(color_predictions, color_targets))
        / len(color_targets)
        if color_targets
        else 0.0
    )
    body_high_confidence = [
        index
        for index, (prediction, confidence) in enumerate(
            zip(body_predictions, body_confidences)
        )
        if prediction != body_unknown_index and confidence >= type_threshold
    ]
    color_high_confidence = [
        index
        for index, (prediction, confidence) in enumerate(
            zip(color_predictions, color_confidences)
        )
        if prediction != color_unknown_index and confidence >= color_threshold
    ]
    body_high_confidence_precision = (
        sum(
            body_predictions[index] == body_targets[index]
            for index in body_high_confidence
        )
        / len(body_high_confidence)
        if body_high_confidence
        else 0.0
    )
    body_family_accuracy = (
        sum(
            truck_family_match(prediction, target, body_hierarchy)
            if body_hierarchy is not None
            else prediction == target
            for prediction, target in zip(body_predictions, body_targets)
        )
        / len(body_targets)
        if body_targets
        else 0.0
    )
    body_family_high_confidence_precision = (
        sum(
            truck_family_match(body_predictions[index], body_targets[index], body_hierarchy)
            if body_hierarchy is not None
            else body_predictions[index] == body_targets[index]
            for index in body_high_confidence
        )
        / len(body_high_confidence)
        if body_high_confidence
        else 0.0
    )
    color_high_confidence_precision = (
        sum(
            color_predictions[index] == color_targets[index]
            for index in color_high_confidence
        )
        / len(color_high_confidence)
        if color_high_confidence
        else 0.0
    )
    return {
        "body_type_accuracy": body_accuracy,
        "body_type_macro_f1": macro_f1(
            body_predictions, body_targets, body_count
        ),
        "color_accuracy": color_accuracy,
        "color_macro_f1": macro_f1(
            color_predictions, color_targets, color_count
        ),
        "body_type_evaluated": len(body_targets),
        "color_evaluated": len(color_targets),
        "body_type_high_confidence_precision": body_high_confidence_precision,
        "body_type_family_accuracy": body_family_accuracy,
        "body_type_family_high_confidence_precision": body_family_high_confidence_precision,
        "body_type_high_confidence_coverage": (
            len(body_high_confidence) / len(body_targets) if body_targets else 0.0
        ),
        "body_type_high_confidence_selected": len(body_high_confidence),
        "color_high_confidence_precision": color_high_confidence_precision,
        "color_high_confidence_coverage": (
            len(color_high_confidence) / len(color_targets) if color_targets else 0.0
        ),
        "color_high_confidence_selected": len(color_high_confidence),
    }


def class_weights(targets: list[int], class_count: int, device):
    import torch

    valid_targets = [int(target) for target in targets if float(target) >= 0]
    counts = torch.bincount(
        torch.tensor(valid_targets, dtype=torch.long),
        minlength=class_count,
    ).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].rsqrt()
    if present.any():
        weights[present] /= weights[present].mean()
    return weights.to(device)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def color_sampling_multipliers(
    rows: list[dict[str, str]],
    colors: list[str],
    mode: str,
    cap: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return bounded train-only class multipliers without fabricating rows."""
    if mode not in {"none", "inverse_sqrt"}:
        raise ValueError(f"unsupported color sampler class weighting: {mode}")
    if cap < 1.0:
        raise ValueError("color sampler class-weight cap must be >= 1")
    known = {color for color in colors if color != "unknown"}
    counts = {color: 0 for color in sorted(known)}
    for row in rows:
        supervised = str(row.get("color_supervised", "true")).strip().lower() not in {
            "false", "0", "no"
        }
        label = str(row.get("color", "")).strip().lower()
        if supervised and label in counts:
            counts[label] += 1
    present = [count for count in counts.values() if count > 0]
    if mode == "none" or not present:
        return {color: 1.0 for color in counts}, counts
    largest = max(present)
    multipliers = {
        color: min(cap, math.sqrt(largest / count)) if count > 0 else 1.0
        for color, count in counts.items()
    }
    return multipliers, counts


def load_fail_closed_unlabeled_rows(manifest: Path) -> list[dict[str, str]]:
    """Load only explicit train/unknown/unsupervised rows.

    The consistency stream never turns a model prediction into a class target.
    A mislabeled or evaluation row therefore fails closed before any image is
    opened.
    """

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("unlabeled manifest must contain train rows only")
        if truthy(row.get("body_type_supervised")) or truthy(row.get("color_supervised")):
            raise ValueError("unlabeled manifest contains a supervised attribute row")
        if row.get("body_type") not in {"", "unknown"} or row.get("color") not in {"", "unknown"}:
            raise ValueError("unlabeled manifest contains a known attribute target")
        if not row.get("image_path"):
            raise ValueError("unlabeled manifest row is missing image_path")
        selected.append(row)
    if not selected:
        raise ValueError("unlabeled manifest has no fail-closed train rows")
    return selected


def resolve_unlabeled_image_path(manifest: Path, safety_root: Path, image_path: str) -> Path:
    """Resolve a manifest-relative image path and enforce a separate safety boundary.

    Multi-source manifests may use ``..`` to reference sibling datasets.  The
    path is therefore interpreted relative to the manifest directory, while
    ``safety_root`` is used only to prevent escaping the approved dataset tree.
    """

    root = safety_root.resolve()
    resolved = (manifest.resolve().parent / image_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unlabeled image escapes safety root: {resolved} not under {root}") from error
    return resolved


def validate_teacher_configuration(args: argparse.Namespace) -> str:
    """Validate legacy or head-specific teacher wiring before importing PyTorch."""

    legacy = args.teacher_checkpoint is not None
    body_specific = args.body_teacher_checkpoint is not None
    color_specific = args.color_teacher_checkpoint is not None
    if legacy and (body_specific or color_specific):
        raise ValueError("--teacher-checkpoint cannot be combined with head-specific teachers")
    enabled = legacy or body_specific or color_specific
    if enabled and args.distill_weight <= 0:
        raise ValueError("teacher distillation requires --distill-weight > 0")
    if args.distill_body_weight < 0 or args.distill_color_weight < 0:
        raise ValueError("distillation head weights must be non-negative")
    if legacy and args.distill_body_weight == 0 and args.distill_color_weight == 0:
        raise ValueError("teacher distillation requires at least one positive head weight")
    if body_specific and args.distill_body_weight <= 0:
        raise ValueError("--body-teacher-checkpoint requires --distill-body-weight > 0")
    if color_specific and args.distill_color_weight <= 0:
        raise ValueError("--color-teacher-checkpoint requires --distill-color-weight > 0")
    if not legacy and not body_specific and color_specific and args.distill_body_weight != 0:
        raise ValueError("color-only teacher distillation requires --distill-body-weight 0")
    if not legacy and body_specific and not color_specific and args.distill_color_weight != 0:
        raise ValueError("body-only teacher distillation requires --distill-color-weight 0")
    return "legacy_multitask" if legacy else "head_specific" if enabled else "disabled"


def forward_teacher_logits(images, teacher=None, body_teacher=None, color_teacher=None):
    """Return each teacher head from its contracted source without cross-head reuse."""

    if teacher is not None:
        return teacher(images)
    if body_teacher is not None and body_teacher is color_teacher:
        return body_teacher(images)
    body_logits = body_teacher(images)[0] if body_teacher is not None else None
    color_logits = color_teacher(images)[1] if color_teacher is not None else None
    return body_logits, color_logits


def main() -> int:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.unlabeled_manifest is None and args.unlabeled_consistency_weight > 0:
        raise ValueError("--unlabeled-consistency-weight requires --unlabeled-manifest")
    if args.unlabeled_manifest is not None and args.unlabeled_consistency_weight <= 0:
        raise ValueError("--unlabeled-manifest requires --unlabeled-consistency-weight > 0")
    if args.unlabeled_body_weight < 0 or args.unlabeled_color_weight < 0:
        raise ValueError("unlabeled head weights must be non-negative")
    teacher_mode = validate_teacher_configuration(args)
    if args.unlabeled_night_sample_weight <= 0:
        raise ValueError("--unlabeled-night-sample-weight must be positive")
    if not 0.0 <= args.unlabeled_ema_decay < 1.0:
        raise ValueError("--unlabeled-ema-decay must be in [0, 1)")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchvision are required") from exc

    from src.attribute_dataset import VehicleAttributeDataset
    from src.multitask_mobilenet_v3 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        MultiTaskVehicleAttributes,
        architecture_from_checkpoint,
        model_from_checkpoint,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labels = load_json(args.labels)
    body_types, colors = trainable_attribute_labels(labels)
    if not 0.0 < args.truck_subtype_threshold <= 1.0:
        raise ValueError("--truck-subtype-threshold must be in (0, 1]")
    if args.coarse_car_loss_weight < 0:
        raise ValueError("--coarse-car-loss-weight must be non-negative")
    if args.coarse_truck_loss_weight < 0:
        raise ValueError("--coarse-truck-loss-weight must be non-negative")
    if args.coarse_color_loss_weight < 0:
        raise ValueError("--coarse-color-loss-weight must be non-negative")
    if args.night_sample_weight <= 0 or args.occlusion_sample_weight <= 0:
        raise ValueError("supervised scene sample weights must be positive")
    body_hierarchy_spec = (
        truck_hierarchy(body_types) if args.body_hierarchy == "truck_family" else None
    )
    coarse_car_indices = (
        car_family_indices(body_types) if args.coarse_car_loss_weight > 0 else None
    )
    coarse_truck_indices = (
        coarse_truck_family_indices(body_types)
        if args.coarse_truck_loss_weight > 0
        else None
    )
    coarse_color_groups = (
        coarse_color_group_indices(colors)
        if args.coarse_color_loss_weight > 0
        else None
    )
    if args.resize_mode == "stretch":
        geometry_transform = transforms.Resize(
            (args.input_size, args.input_size),
            antialias=True,
        )
        train_geometry_transform = geometry_transform
        evaluation_geometry_transforms = [geometry_transform]
    elif args.resize_mode == "center_crop":
        resize_size = round(args.input_size * 232 / 224)
        train_geometry_transform = transforms.RandomResizedCrop(
            args.input_size,
            scale=(0.75, 1.0),
        )
        evaluation_geometry_transforms = [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(args.input_size),
        ]
    else:
        square_pad = SquarePad()
        resize = transforms.Resize(
            (args.input_size, args.input_size),
            antialias=True,
        )
        train_geometry_transform = transforms.Compose([square_pad, resize])
        evaluation_geometry_transforms = [square_pad, resize]
    train_ops = [
        train_geometry_transform,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
    ]
    if args.augmentation_profile == "hard_scene":
        train_ops.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.04)],
                    p=0.65,
                ),
                transforms.RandomAutocontrast(p=0.25),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6))], p=0.20),
                transforms.RandomGrayscale(p=0.04),
                RandomShadowHighlight(probability=0.25),
                RandomJPEGCompression(probability=0.25, minimum_quality=45, maximum_quality=90),
            ]
        )
    elif args.augmentation_profile == "color_scene":
        train_ops.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.35, contrast=0.30, saturation=0.20, hue=0.025)],
                    p=0.65,
                ),
                transforms.RandomAutocontrast(p=0.15),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
                RandomColorTemperature(probability=0.35, maximum_shift=0.08),
                RandomShadowHighlight(probability=0.35),
                RandomJPEGCompression(probability=0.30, minimum_quality=50, maximum_quality=92),
            ]
        )
    else:
        train_ops.append(transforms.ColorJitter(brightness=0.20, contrast=0.20))
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_transform = transforms.Compose(train_ops)
    evaluation_transform = transforms.Compose(
        [
            *evaluation_geometry_transforms,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    unlabeled_loader = None
    unlabeled_rows: list[dict[str, str]] = []
    unlabeled_manifest_sha256 = None
    unlabeled_night_rows = 0
    unlabeled_expected_night_fraction = 0.0
    if args.unlabeled_manifest is not None:
        unlabeled_manifest = args.unlabeled_manifest.resolve()
        unlabeled_root = (
            args.unlabeled_root.resolve()
            if args.unlabeled_root is not None
            else unlabeled_manifest.parent.resolve()
        )
        unlabeled_rows = load_fail_closed_unlabeled_rows(unlabeled_manifest)
        unlabeled_manifest_sha256 = sha256_file(unlabeled_manifest)
        weak_transform = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        strong_transform = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size), antialias=True),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.45, contrast=0.40, saturation=0.20, hue=0.025)],
                    p=0.85,
                ),
                transforms.RandomAutocontrast(p=0.20),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6))], p=0.25),
                RandomColorTemperature(probability=0.30, maximum_shift=0.08),
                RandomShadowHighlight(probability=0.35),
                RandomJPEGCompression(probability=0.35, minimum_quality=45, maximum_quality=90),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        class UnlabeledPairDataset(Dataset):
            def __len__(self) -> int:
                return len(unlabeled_rows)

            def __getitem__(self, index: int):
                from PIL import Image

                row = unlabeled_rows[index]
                path = resolve_unlabeled_image_path(
                    unlabeled_manifest,
                    unlabeled_root,
                    row["image_path"],
                )
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                return weak_transform(image), strong_transform(image)

        unlabeled_night_rows = sum(truthy(row.get("night")) for row in unlabeled_rows)
        unlabeled_weights = [
            args.unlabeled_night_sample_weight if truthy(row.get("night")) else 1.0
            for row in unlabeled_rows
        ]
        weighted_night = unlabeled_night_rows * args.unlabeled_night_sample_weight
        weighted_total = weighted_night + len(unlabeled_rows) - unlabeled_night_rows
        unlabeled_expected_night_fraction = weighted_night / weighted_total
        unlabeled_sampler = WeightedRandomSampler(
            unlabeled_weights,
            len(unlabeled_weights),
            replacement=True,
        )
        unlabeled_loader = DataLoader(
            UnlabeledPairDataset(),
            batch_size=args.unlabeled_batch_size,
            sampler=unlabeled_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
    train_dataset = VehicleAttributeDataset(
        args.manifest,
        split="train",
        body_types=body_types,
        colors=colors,
        transform=train_transform,
        training=True,
        return_pseudo=True,
        return_body_family=coarse_car_indices is not None or coarse_truck_indices is not None,
        return_color_group=coarse_color_groups is not None,
    )
    validation_dataset = VehicleAttributeDataset(
        args.manifest,
        split="validation",
        body_types=body_types,
        colors=colors,
        transform=evaluation_transform,
        training=False,
    )
    test_dataset = None if args.skip_test else VehicleAttributeDataset(
        args.manifest,
        split="test",
        body_types=body_types,
        colors=colors,
        transform=evaluation_transform,
        training=False,
    )
    color_sampler_multipliers, color_sampler_counts = color_sampling_multipliers(
        train_dataset.rows,
        colors,
        args.sampler_color_class_weighting,
        args.sampler_color_class_weight_cap,
    )
    sampler = None
    if (
        args.hard_sample_weight > 0
        or args.small_sample_weight > 0
        or args.color_sample_weight > 0
        or args.night_sample_weight != 1.0
        or args.occlusion_sample_weight != 1.0
        or args.sampler_color_class_weighting != "none"
    ):
        sample_weights = []
        for row in train_dataset.rows:
            try:
                score = max(0.0, float(row.get("hard_score", "0") or 0.0))
            except ValueError:
                score = 0.0
            weight = 1.0 + args.hard_sample_weight * min(score, 8.0) / 8.0
            try:
                explicit_weight = float(row.get("sample_weight", "1") or 1.0)
            except ValueError:
                explicit_weight = 1.0
            weight *= max(0.10, min(10.0, explicit_weight))
            small_marked = str(row.get("small_target", "")).strip().lower() in {"1", "true", "yes"}
            small_marked = small_marked or str(row.get("vehicle_size", "")).strip().lower() == "small"
            if small_marked and args.small_sample_weight > 0:
                weight *= 1.0 + args.small_sample_weight
            color_value = str(row.get("color", "")).strip().lower()
            color_supervised = str(row.get("color_supervised", "true")).strip().lower() not in {"false", "0", "no"}
            if color_supervised and color_value not in {"", "unknown"} and args.color_sample_weight > 0:
                weight *= 1.0 + args.color_sample_weight
            if color_supervised and color_value in color_sampler_multipliers:
                weight *= color_sampler_multipliers[color_value]
            if truthy(row.get("night")):
                weight *= args.night_sample_weight
            if truthy(row.get("occluded")) or truthy(row.get("truncated")):
                weight *= args.occlusion_sample_weight
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = None if test_dataset is None else DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskVehicleAttributes(
        len(body_types),
        len(colors),
        architecture=args.architecture,
        pretrained=args.resume is None and args.init_checkpoint is None,
    ).to(device)
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        if architecture_from_checkpoint(init_checkpoint) != args.architecture:
            raise ValueError("--init-checkpoint architecture does not match --architecture")
        model.load_state_dict(init_checkpoint["model_state"])
    teacher_cache = {}

    def load_frozen_teacher(path: Path | None):
        if path is None:
            return None
        resolved = path.resolve()
        if resolved in teacher_cache:
            return teacher_cache[resolved]
        teacher_checkpoint = torch.load(resolved, map_location="cpu")
        frozen = model_from_checkpoint(teacher_checkpoint, pretrained=False).to(device)
        frozen.load_state_dict(teacher_checkpoint["model_state"])
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad = False
        teacher_cache[resolved] = frozen
        return frozen

    teacher = load_frozen_teacher(args.teacher_checkpoint)
    body_teacher = load_frozen_teacher(args.body_teacher_checkpoint)
    color_teacher = load_frozen_teacher(args.color_teacher_checkpoint)
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs must be non-negative")
    if args.freeze_backbone_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    start_epoch = 0
    best_score = -1.0
    gate_best_key = (-1.0, -1.0, -1.0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if architecture_from_checkpoint(checkpoint) != args.architecture:
            raise ValueError("--resume checkpoint architecture does not match --architecture")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        gate_best_key = tuple(
            float(value) for value in checkpoint.get("gate_best_key", gate_best_key)
        )
        resumed_epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
    else:
        resumed_epochs_without_improvement = 0

    # A resumed run can start after the warm-up freeze boundary. In that case,
    # the epoch-loop equality check below will never fire, so restore the
    # backbone's trainable state before the resumed optimizer steps begin.
    if args.freeze_backbone_epochs and start_epoch >= args.freeze_backbone_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True

    consistency_teacher = None
    if unlabeled_loader is not None:
        consistency_teacher = copy.deepcopy(model).to(device).eval()
        for parameter in consistency_teacher.parameters():
            parameter.requires_grad = False

    body_targets, color_targets = train_dataset.targets()
    body_weights = (
        class_weights(body_targets, len(body_types), device)
        if args.class_weighting == "inverse_sqrt"
        else None
    )
    color_weights = (
        class_weights(color_targets, len(colors), device)
        if args.class_weighting == "inverse_sqrt"
        else None
    )
    body_criterion = nn.CrossEntropyLoss(
        weight=body_weights,
        label_smoothing=args.label_smoothing,
        reduction="none",
    )
    color_criterion = nn.CrossEntropyLoss(
        weight=color_weights,
        label_smoothing=args.label_smoothing,
        reduction="none",
    )

    def head_loss(criterion, logits, targets, gamma: float, sample_weights=None):
        values = criterion(logits, targets)
        if gamma > 0:
            with torch.no_grad():
                pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
            values = values * (1.0 - pt).clamp_min(1e-6).pow(gamma)
        if sample_weights is not None:
            values = values * sample_weights
            return values.sum() / sample_weights.sum().clamp_min(1e-6)
        return values.mean()
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    epochs_without_improvement = resumed_epochs_without_improvement
    started = datetime.now(timezone.utc)
    latest_metrics: dict[str, Any] = {}

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
        model.train()
        if epoch < args.freeze_backbone_epochs:
            model.backbone.eval()
        running_loss = 0.0
        running_unlabeled_consistency = 0.0
        trained_batches = 0
        unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None else None
        for batch in train_loader:
            if len(batch) == 7:
                (
                    images,
                    body_target,
                    color_target,
                    _,
                    pseudo_label,
                    coarse_body_family,
                    coarse_color_group,
                ) = batch
            elif len(batch) == 6:
                images, body_target, color_target, _, pseudo_label, coarse_body_family = batch
                coarse_color_group = torch.zeros_like(color_target)
            elif len(batch) == 5:
                images, body_target, color_target, _, pseudo_label = batch
                coarse_body_family = torch.zeros_like(body_target)
                coarse_color_group = torch.zeros_like(color_target)
            else:
                images, body_target, color_target, _ = batch
                pseudo_label = torch.zeros_like(body_target, dtype=torch.bool)
                coarse_body_family = torch.zeros_like(body_target)
                coarse_color_group = torch.zeros_like(color_target)
            images = images.to(device, non_blocking=True)
            body_target = body_target.to(device, non_blocking=True)
            color_target = color_target.to(device, non_blocking=True)
            coarse_body_family = coarse_body_family.to(device, non_blocking=True)
            coarse_color_group = coarse_color_group.to(device, non_blocking=True)
            pseudo_weight = torch.where(
                pseudo_label.to(device, non_blocking=True),
                torch.full_like(body_target, float(args.pseudo_label_weight), dtype=torch.float32),
                torch.ones_like(body_target, dtype=torch.float32),
            )
            unlabeled_pair = None
            if unlabeled_iterator is not None:
                try:
                    unlabeled_pair = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    unlabeled_pair = next(unlabeled_iterator)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                body_logits, color_logits = model(images)
                loss = torch.zeros((), device=device)
                valid_heads = 0
                body_mask = body_target != -100
                coarse_car_mask = coarse_body_family == 1
                coarse_truck_mask = coarse_body_family == 2
                color_mask = color_target != -100
                coarse_color_mask = coarse_color_group != 0
                if body_mask.any() or coarse_car_mask.any() or coarse_truck_mask.any():
                    body_loss = torch.zeros((), device=device)
                    if body_hierarchy_spec is None:
                        if body_mask.any():
                            body_loss = head_loss(
                                body_criterion,
                                body_logits[body_mask],
                                body_target[body_mask],
                                args.focal_gamma,
                                pseudo_weight[body_mask],
                            )
                    else:
                        if body_mask.any():
                            body_values = body_criterion(
                                body_logits[body_mask], body_target[body_mask]
                            )
                            if args.focal_gamma > 0:
                                with torch.no_grad():
                                    body_pt = torch.softmax(body_logits[body_mask], dim=1).gather(
                                        1, body_target[body_mask].unsqueeze(1)
                                    ).squeeze(1)
                                body_values = body_values * (
                                    1.0 - body_pt
                                ).clamp_min(1e-6).pow(args.focal_gamma)
                            body_loss = partial_truck_family_loss(
                                body_logits[body_mask],
                                body_target[body_mask],
                                body_hierarchy_spec,
                                body_values,
                                pseudo_weight[body_mask],
                                coarse_class_weight=(
                                    body_weights[body_hierarchy_spec.generic]
                                    if body_weights is not None
                                    else None
                                ),
                                gamma=args.focal_gamma,
                            )
                    if coarse_car_mask.any():
                        body_loss = body_loss + args.coarse_car_loss_weight * partial_coarse_family_loss(
                            body_logits[coarse_car_mask],
                            coarse_car_indices,
                            pseudo_weight[coarse_car_mask],
                            gamma=args.focal_gamma,
                        )
                    if coarse_truck_mask.any():
                        body_loss = body_loss + args.coarse_truck_loss_weight * partial_coarse_family_loss(
                            body_logits[coarse_truck_mask],
                            coarse_truck_indices,
                            pseudo_weight[coarse_truck_mask],
                            gamma=args.focal_gamma,
                        )
                    loss = loss + args.body_loss_weight * body_loss
                    valid_heads += 1
                if color_mask.any() or coarse_color_mask.any():
                    color_loss = torch.zeros((), device=device)
                    if color_mask.any():
                        color_loss = head_loss(
                            color_criterion,
                            color_logits[color_mask],
                            color_target[color_mask],
                            args.color_focal_gamma,
                            pseudo_weight[color_mask],
                        )
                    if coarse_color_mask.any():
                        color_loss = color_loss + args.coarse_color_loss_weight * partial_label_group_loss(
                            color_logits[coarse_color_mask],
                            coarse_color_group[coarse_color_mask],
                            coarse_color_groups,
                            pseudo_weight[coarse_color_mask],
                            gamma=args.color_focal_gamma,
                        )
                    loss = loss + args.color_loss_weight * color_loss
                    valid_heads += 1
                if teacher is not None or body_teacher is not None or color_teacher is not None:
                    with torch.no_grad():
                        teacher_body, teacher_color = forward_teacher_logits(
                            images, teacher, body_teacher, color_teacher
                        )
                    temperature = max(1e-3, float(args.distill_temperature))
                    distill = torch.zeros((), device=device)
                    if teacher_body is not None and args.distill_body_weight > 0:
                        body_distill = torch.nn.functional.kl_div(
                            torch.log_softmax(body_logits / temperature, dim=1),
                            torch.softmax(teacher_body / temperature, dim=1),
                            reduction="batchmean",
                        )
                        distill = distill + float(args.distill_body_weight) * body_distill
                    if teacher_color is not None and args.distill_color_weight > 0:
                        color_distill = torch.nn.functional.kl_div(
                            torch.log_softmax(color_logits / temperature, dim=1),
                            torch.softmax(teacher_color / temperature, dim=1),
                            reduction="batchmean",
                        )
                        distill = distill + float(args.distill_color_weight) * color_distill
                    distill = distill * (temperature * temperature)
                    loss = loss + float(args.distill_weight) * distill
                consistency = torch.zeros((), device=device)
                if unlabeled_pair is not None and consistency_teacher is not None:
                    weak_images, strong_images = unlabeled_pair
                    weak_images = weak_images.to(device, non_blocking=True)
                    strong_images = strong_images.to(device, non_blocking=True)
                    if args.unlabeled_consistency_mode == "feature":
                        with torch.no_grad():
                            weak_body, weak_color = consistency_teacher.forward_features(weak_images)
                        strong_body, strong_color = model.forward_features(strong_images)
                        body_consistency = (
                            1.0 - torch.nn.functional.cosine_similarity(
                                strong_body, weak_body.detach(), dim=1
                            )
                        ).mean()
                        color_consistency = (
                            1.0 - torch.nn.functional.cosine_similarity(
                                strong_color, weak_color.detach(), dim=1
                            )
                        ).mean()
                    else:
                        with torch.no_grad():
                            weak_body, weak_color = consistency_teacher(weak_images)
                        strong_body, strong_color = model(strong_images)
                        consistency_temperature = max(
                            1e-3, float(args.unlabeled_consistency_temperature)
                        )
                        body_consistency = torch.nn.functional.kl_div(
                            torch.log_softmax(strong_body / consistency_temperature, dim=1),
                            torch.softmax(weak_body / consistency_temperature, dim=1),
                            reduction="batchmean",
                        ) * (consistency_temperature * consistency_temperature)
                        color_consistency = torch.nn.functional.kl_div(
                            torch.log_softmax(strong_color / consistency_temperature, dim=1),
                            torch.softmax(weak_color / consistency_temperature, dim=1),
                            reduction="batchmean",
                        ) * (consistency_temperature * consistency_temperature)
                    consistency = (
                        float(args.unlabeled_body_weight) * body_consistency
                        + float(args.unlabeled_color_weight) * color_consistency
                    )
                    loss = loss + float(args.unlabeled_consistency_weight) * consistency
            if not valid_heads:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if consistency_teacher is not None:
                decay = float(args.unlabeled_ema_decay)
                with torch.no_grad():
                    for teacher_parameter, model_parameter in zip(
                        consistency_teacher.parameters(), model.parameters()
                    ):
                        teacher_parameter.mul_(decay).add_(
                            model_parameter.detach(), alpha=1.0 - decay
                        )
                    for teacher_buffer, model_buffer in zip(
                        consistency_teacher.buffers(), model.buffers()
                    ):
                        teacher_buffer.copy_(model_buffer)
            running_loss += float(loss.detach().cpu())
            running_unlabeled_consistency += float(consistency.detach().cpu())
            trained_batches += 1

        metrics = evaluate(
            model,
            validation_loader,
            device,
            len(body_types),
            len(colors),
            body_types.index("unknown"),
            colors.index("unknown"),
            args.type_threshold,
            args.color_threshold,
            body_hierarchy_spec,
            args.truck_subtype_threshold,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, trained_batches)
        metrics["unlabeled_consistency_loss"] = (
            running_unlabeled_consistency / max(1, trained_batches)
        )
        metrics["unlabeled_rows"] = len(unlabeled_rows)
        metrics["unlabeled_night_rows"] = unlabeled_night_rows
        metrics["unlabeled_expected_night_fraction"] = unlabeled_expected_night_fraction
        if args.selection_head == "body":
            score = float(metrics["body_type_macro_f1"])
        elif args.selection_head == "color":
            score = float(metrics["color_macro_f1"])
        else:
            score = (
                float(metrics["body_type_macro_f1"])
                + float(metrics["color_macro_f1"])
            ) / 2.0
        metrics["selection_score"] = score
        latest_metrics = metrics
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        improved = score > best_score
        body_gate = (
            float(metrics["body_type_high_confidence_precision"]) >= args.gate_type_precision
            and float(metrics["body_type_high_confidence_coverage"]) >= args.gate_type_coverage
        )
        color_gate = (
            float(metrics["color_high_confidence_precision"]) >= args.gate_color_precision
            and float(metrics["color_high_confidence_coverage"]) >= args.gate_color_coverage
        )
        gate_qualified = (
            body_gate if args.selection_head == "body"
            else color_gate if args.selection_head == "color"
            else body_gate and color_gate
        )
        gate_key = (
            float(metrics["color_high_confidence_coverage"]), score, 0.0
        ) if args.selection_head == "color" else (
            float(metrics["body_type_high_confidence_coverage"]), score, 0.0
        ) if args.selection_head == "body" else (
            float(metrics["body_type_high_confidence_coverage"]),
            float(metrics["color_high_confidence_coverage"]),
            score,
        )
        gate_improved = gate_qualified and gate_key > gate_best_key
        if gate_improved:
            gate_best_key = gate_key
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        checkpoint = {
            "schema_version": "1.0",
            "epoch": epoch,
            "best_score": best_score,
            "gate_best_key": list(gate_best_key),
            "epochs_without_improvement": epochs_without_improvement,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "architecture": f"{args.architecture}_multitask",
            "input_size": args.input_size,
            "resize_mode": args.resize_mode,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "selection_head": args.selection_head,
            "unlabeled_consistency_weight": args.unlabeled_consistency_weight,
            "unlabeled_consistency_mode": args.unlabeled_consistency_mode,
            "unlabeled_consistency_temperature": args.unlabeled_consistency_temperature,
            "unlabeled_ema_decay": args.unlabeled_ema_decay,
            "body_hierarchy": args.body_hierarchy,
            "truck_subtype_threshold": args.truck_subtype_threshold,
            "coarse_car_loss_weight": args.coarse_car_loss_weight,
            "body_types": body_types,
            "colors": colors,
            "labels_version": labels["labels_version"],
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        if gate_improved:
            torch.save(checkpoint, output_dir / "gate-best.pt")
        print(f"epoch={epoch} metrics={metrics}")
        if epochs_without_improvement >= args.patience:
            break

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state"])
    model.to(device)
    test_metrics = (
        {
            "status": "not_run",
            "reason": "--skip-test preserves the independent test split for final evaluation",
        }
        if test_loader is None
        else evaluate(
            model,
            test_loader,
            device,
            len(body_types),
            len(colors),
            body_types.index("unknown"),
            colors.index("unknown"),
            args.type_threshold,
            args.color_threshold,
            body_hierarchy_spec,
            args.truck_subtype_threshold,
        )
    )
    summary = {
        "schema_version": "1.0",
        "run_id": output_dir.name,
        "run_kind": args.run_kind,
        "kind": "attributes",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [sys.executable, *sys.argv],
        "code_revision": args.code_revision,
        "dataset_version": args.dataset_version,
        "architecture": f"{args.architecture}_multitask",
        "input_size": args.input_size,
        "resize_mode": args.resize_mode,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "labels_version": labels["labels_version"],
        "body_types": body_types,
        "colors": colors,
        "unknown_policy": (
            "unknown is an explicit model output; poor crops supervise unknown "
            "for both heads, and runtime thresholds can also reject unstable "
            "known-class predictions"
        ),
        "augmentation_profile": args.augmentation_profile,
        "selection_head": args.selection_head,
        "distillation": {
            "enabled": teacher_mode != "disabled",
            "mode": teacher_mode,
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()) if args.teacher_checkpoint else None,
            "body_teacher_checkpoint": str(args.body_teacher_checkpoint.resolve()) if args.body_teacher_checkpoint else None,
            "color_teacher_checkpoint": str(args.color_teacher_checkpoint.resolve()) if args.color_teacher_checkpoint else None,
            "weight": args.distill_weight,
            "temperature": args.distill_temperature,
            "body_weight": args.distill_body_weight,
            "color_weight": args.distill_color_weight,
        },
        "body_hierarchy": {
            "mode": args.body_hierarchy,
            "truck_subtype_threshold": args.truck_subtype_threshold,
            "generic_truck_loss": (
                "negative log probability mass of truck/light_truck/heavy_truck"
                if body_hierarchy_spec is not None
                else "flat cross entropy"
            ),
            "gate_metric": "exact body_type_high_confidence_precision; family-aware precision is reported separately and never substitutes for the exact gate",
        },
        "hard_sample_weight": args.hard_sample_weight,
        "small_sample_weight": args.small_sample_weight,
        "color_sample_weight": args.color_sample_weight,
        "color_sampler_class_weighting": {
            "mode": args.sampler_color_class_weighting,
            "cap": args.sampler_color_class_weight_cap,
            "supervised_train_counts": color_sampler_counts,
            "multipliers": color_sampler_multipliers,
            "policy": "bounded train-only weighted sampling; no files or labels are duplicated",
        },
        "night_sample_weight": args.night_sample_weight,
        "occlusion_sample_weight": args.occlusion_sample_weight,
        "coarse_car_partial_supervision": {
            "enabled": coarse_car_indices is not None,
            "loss_weight": args.coarse_car_loss_weight,
            "family_labels": [body_types[index] for index in coarse_car_indices] if coarse_car_indices else [],
            "policy": "official coarse car truth supervises family probability mass and never becomes a fabricated exact class",
        },
        "coarse_truck_partial_supervision": {
            "enabled": coarse_truck_indices is not None,
            "loss_weight": args.coarse_truck_loss_weight,
            "family_labels": [body_types[index] for index in coarse_truck_indices] if coarse_truck_indices else [],
            "policy": "official coarse truck truth supervises available truck-family probability mass and never becomes a fabricated exact subtype",
        },
        "coarse_color_partial_supervision": {
            "enabled": coarse_color_groups is not None,
            "loss_weight": args.coarse_color_loss_weight,
            "group_labels": {
                str(code): [colors[index] for index in indices]
                for code, indices in (coarse_color_groups or {}).items()
            },
            "policy": (
                "legacy merged color truth supervises only the compatible v2 probability mass; "
                "silver_gray never becomes a fabricated exact gray or silver label"
            ),
        },
        "manifest_sample_weight": "sample_weight column, clamped to [0.10, 10.0]",
        "pseudo_label_weight": args.pseudo_label_weight,
        "unlabeled_domain_consistency": {
            "enabled": unlabeled_loader is not None,
            "manifest": str(args.unlabeled_manifest.resolve()) if args.unlabeled_manifest else None,
            "manifest_sha256": unlabeled_manifest_sha256,
            "image_root": str(
                args.unlabeled_root.resolve()
                if args.unlabeled_root is not None
                else args.unlabeled_manifest.resolve().parent
            ) if args.unlabeled_manifest else None,
            "rows": len(unlabeled_rows),
            "night_rows": unlabeled_night_rows,
            "expected_sampled_night_fraction": unlabeled_expected_night_fraction,
            "consistency_weight": args.unlabeled_consistency_weight,
            "consistency_mode": args.unlabeled_consistency_mode,
            "body_weight": args.unlabeled_body_weight,
            "color_weight": args.unlabeled_color_weight,
            "temperature": args.unlabeled_consistency_temperature,
            "ema_decay": args.unlabeled_ema_decay,
            "label_policy": (
                "every row must be train-only, unknown and unsupervised; EMA weak-view "
                + (
                    "pre-head embeddings supervise normalized strong-view feature consistency "
                    "without propagating any class or unknown decision"
                    if args.unlabeled_consistency_mode == "feature"
                    else "distributions supervise strong-view consistency without creating class labels"
                )
            ),
        },
        "color_augmentation": (
            "hard_scene: brightness/contrast/saturation/hue, autocontrast, blur, grayscale, soft shadow/reflection and JPEG compression"
            if args.augmentation_profile == "hard_scene"
            else "color_scene: bounded brightness/contrast/saturation/hue and color temperature, autocontrast, blur, soft shadow/reflection and JPEG compression; grayscale disabled"
            if args.augmentation_profile == "color_scene"
            else "brightness/contrast only; saturation and hue changes are disabled"
        ),
        "preprocessing": (
            "stretch to square, then ImageNet normalization"
            if args.resize_mode == "stretch"
            else "resize and center crop, then ImageNet normalization"
            if args.resize_mode == "center_crop"
            else "center-pad to square, resize, then ImageNet normalization"
        ),
        "optimization": {
            "class_weighting": args.class_weighting,
            "body_loss_weight": args.body_loss_weight,
            "focal_gamma": args.focal_gamma,
            "color_focal_gamma": args.color_focal_gamma,
            "label_smoothing": args.label_smoothing,
            "gradient_clip_norm": args.gradient_clip_norm,
            "scheduler": "cosine_annealing",
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
        },
        "runtime_thresholds": {
            "body_type": args.type_threshold,
            "color": args.color_threshold,
            "high_confidence_precision_target": 0.93,
        },
        "metrics": {
            "validation": latest_metrics,
            "test": test_metrics,
        },
        "artifacts": {
            "best_pt": str(best_path),
            "best_pt_sha256": sha256_file(best_path),
            "last_pt": str(last_path),
            "last_pt_sha256": sha256_file(last_path),
        },
        "release_eligible": False,
        "release_status": (
            "candidate_pending_conversion_and_release_validation"
            if args.run_kind == "formal"
            else "smoke_only"
        ),
        "note": (
            "Formal training candidate; release still requires threshold calibration, "
            "ONNX/TensorRT regression, model-card review, and release acceptance."
            if args.run_kind == "formal"
            else "Smoke artifact only; export and ONNX validation are still required."
        ),
    }
    write_json(output_dir / "metrics.json", latest_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_json(output_dir / "model_card.json", summary)
    print(f"PASS: attribute {args.run_kind} run complete: {output_dir / 'model_card.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
