#!/usr/bin/env python3
"""Evaluate a VCAS attribute checkpoint with stratified baseline evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def derive_metadata(row: dict[str, str], image_path: Path | None = None) -> dict[str, str]:
    truncated_raw = (row.get("truncated") or "").strip()
    occluded_raw = (row.get("occluded") or "").strip()
    if not truncated_raw and not occluded_raw:
        occlusion = "unknown"
    elif parse_bool(truncated_raw):
        occlusion = "truncated"
    elif parse_bool(row.get("occluded")):
        occlusion = "occluded"
    else:
        occlusion = "visible"
    try:
        area = float(row.get("width", "")) * float(row.get("height", ""))
    except (TypeError, ValueError):
        area = 0.0
    gray_from_image: float | None = None
    if (area <= 0 or not (row.get("gray_mean") or "").strip()) and image_path and image_path.exists():
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
                area = float(width * height)
                gray_from_image = float(image.convert("L").resize((1, 1)).getpixel((0, 0)))
        except Exception:
            pass
    explicit_size = str(row.get("vehicle_size", "")).strip().lower()
    if str(row.get("small_target", "")).strip().lower() in {"1", "true", "yes"}:
        explicit_size = "small"
    size = explicit_size if explicit_size in {"small", "medium", "large"} else ("unknown" if area <= 0 else "small" if area < 128 * 96 else "medium" if area < 256 * 192 else "large")
    if parse_bool(row.get("night")):
        lighting = "night"
    else:
        try:
            mean = float(row.get("gray_mean", ""))
        except (TypeError, ValueError):
            mean = math.nan if gray_from_image is None else gray_from_image
        lighting = "unknown" if math.isnan(mean) else "low_light" if mean < 64 else "moderate_light" if mean < 128 else "daylight"
    review = (row.get("review_status") or "").strip().lower()
    body = (row.get("body_type") or "").strip().lower()
    color = (row.get("color") or "").strip().lower()
    confidence = "high" if review == "approved" and (body not in {"", "unknown"} or color not in {"", "unknown"}) else "medium" if review == "approved" else "unknown"
    return {
        "crop_quality": row.get("crop_quality") or "unknown",
        "occlusion_level": occlusion,
        "vehicle_size": size,
        "lighting": lighting,
        "viewpoint": row.get("viewpoint") or "unknown",
        "source_dataset": row.get("source_dataset") or "unknown",
        "label_confidence": confidence,
    }


def macro_f1(pred: list[int], target: list[int], class_count: int) -> float:
    values = []
    for class_id in range(class_count):
        tp = sum(p == class_id and t == class_id for p, t in zip(pred, target))
        fp = sum(p == class_id and t != class_id for p, t in zip(pred, target))
        fn = sum(p != class_id and t == class_id for p, t in zip(pred, target))
        den = 2 * tp + fp + fn
        if den:
            values.append(2 * tp / den)
    return sum(values) / len(values) if values else 0.0


def head_metrics(pred: list[int], target: list[int], confidence: list[float], labels: list[str], threshold: float) -> dict:
    unknown = labels.index("unknown") if "unknown" in labels else -1
    selected = [i for i, (p, c) in enumerate(zip(pred, confidence)) if p != unknown and c >= threshold]
    return {
        "evaluated": len(target),
        "accuracy": sum(p == t for p, t in zip(pred, target)) / len(target) if target else 0.0,
        "macro_f1": macro_f1(pred, target, len(labels)),
        "high_confidence_precision": sum(pred[i] == target[i] for i in selected) / len(selected) if selected else 0.0,
        "high_confidence_coverage": len(selected) / len(target) if target else 0.0,
        "high_confidence_selected": len(selected),
        "predicted_unknown_rate": sum(p == unknown for p in pred) / len(pred) if pred else 0.0,
        "true_unknown_rate": sum(t == unknown for t in target) / len(target) if target else 0.0,
        "predicted_counts": dict(sorted(Counter(labels[p] for p in pred).items())),
        "true_counts": dict(sorted(Counter(labels[t] for t in target).items())),
        "confusion": {
            labels[t]: dict(sorted(Counter(labels[p] for p, tt in zip(pred, target) if tt == t).items()))
            for t in sorted(set(target))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--type-threshold", type=float, default=0.75)
    parser.add_argument("--color-threshold", type=float, default=0.70)
    parser.add_argument("--body-temperature", type=float, default=1.0)
    parser.add_argument("--color-temperature", type=float, default=1.0)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="*",
        default=[],
        help="Optional confidence thresholds evaluated from the same inference pass.",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import sys
    candidates = [
        Path(__file__).resolve().parents[1],
        args.manifest.parents[2] / "code" / "training",
        args.manifest.parents[2] / "training",
    ]
    for candidate in candidates:
        if (candidate / "src").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError("could not locate VCAS training/src on PYTHONPATH")
    from src.common import trainable_attribute_labels, load_json
    from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD
    try:
        from src.multitask_mobilenet_v3 import model_from_checkpoint
    except ImportError:
        model_from_checkpoint = None

    labels_config = load_json(args.labels)
    body_labels, color_labels = trainable_attribute_labels(labels_config)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("split") == args.split]
    rows = [r for r in rows if (r.get("review_status", "approved") or "approved") == "approved"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # The deployed checkpoint is authoritative for output order.  Older cloud
    # source trees may have a label helper that omits the explicit unknown
    # class even though the canonical checkpoint has ten outputs.
    body_labels = [str(x) for x in checkpoint.get("body_types", body_labels)]
    color_labels = [str(x) for x in checkpoint.get("colors", color_labels)]
    input_size = int(checkpoint.get("input_size", 224))
    transform = transforms.Compose([transforms.Resize((input_size, input_size), antialias=True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class Rows(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            with Image.open(args.manifest.parent / row["image_path"]) as image:
                image = image.convert("RGB")
            body = body_labels.index(row["body_type"]) if row.get("body_type") in body_labels and (row.get("body_type_supervised", "true").lower() not in {"false", "0", "no"}) else -100
            color = color_labels.index(row["color"]) if row.get("color") in color_labels and (row.get("color_supervised", "true").lower() not in {"false", "0", "no"}) else -100
            return transform(image), body, color, index

    loader = DataLoader(Rows(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if model_from_checkpoint is not None:
        model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
    else:
        from src.multitask_mobilenet_v3 import MultiTaskMobileNetV3
        model = MultiTaskMobileNetV3(len(body_labels), len(color_labels), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    body_values: list[tuple[int, int, float, int]] = []
    color_values: list[tuple[int, int, float, int]] = []
    with torch.no_grad():
        for images, body_target, color_target, indexes in loader:
            body_logits, color_logits = model(images.to(device))
            bp = (body_logits / args.body_temperature).softmax(1); cp = (color_logits / args.color_temperature).softmax(1)
            bc, bi = bp.max(1); cc, ci = cp.max(1)
            for t, p, c, i in zip(body_target.tolist(), bi.tolist(), bc.tolist(), indexes.tolist()):
                if t != -100: body_values.append((p, t, c, i))
            for t, p, c, i in zip(color_target.tolist(), ci.tolist(), cc.tolist(), indexes.tolist()):
                if t != -100: color_values.append((p, t, c, i))

    def stratified(values: list[tuple[int, int, float, int]], labels: list[str], threshold: float) -> dict:
        groups: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        for p, t, c, i in values:
            meta = derive_metadata(rows[i], args.manifest.parent / rows[i]["image_path"])
            for key, value in meta.items(): groups[f"{key}={value}"].append((p, t, c))
        return {key: head_metrics([x[0] for x in vals], [x[1] for x in vals], [x[2] for x in vals], labels, threshold) for key, vals in sorted(groups.items())}

    body = [(p, t, c) for p, t, c, _ in body_values]
    color = [(p, t, c) for p, t, c, _ in color_values]
    report = {
        "schema_version": "1.0",
        "protocol": {"fixed_video_used_for_selection": False, "test_used_for_threshold_or_model_selection": False, "thresholds": {"body_type": args.type_threshold, "color": args.color_threshold}, "temperatures": {"body_type": args.body_temperature, "color": args.color_temperature}},
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "rows": len(rows),
        "body_type": head_metrics([x[0] for x in body], [x[1] for x in body], [x[2] for x in body], body_labels, args.type_threshold),
        "color": head_metrics([x[0] for x in color], [x[1] for x in color], [x[2] for x in color], color_labels, args.color_threshold),
        "stratified": {"body_type": stratified(body_values, body_labels, args.type_threshold), "color": stratified(color_values, color_labels, args.color_threshold)},
        "unknown_distribution": {"body_type": dict(sorted(Counter(body_labels[p] for p, _, _, _ in body_values).items())), "color": dict(sorted(Counter(color_labels[p] for p, _, _, _ in color_values).items()))},
        "threshold_sweep": {
            "body_type": {
                f"{threshold:.4f}": head_metrics(
                    [x[0] for x in body],
                    [x[1] for x in body],
                    [x[2] for x in body],
                    body_labels,
                    threshold,
                )
                for threshold in sorted(set(args.threshold_sweep))
            },
            "color": {
                f"{threshold:.4f}": head_metrics(
                    [x[0] for x in color],
                    [x[1] for x in color],
                    [x[2] for x in color],
                    color_labels,
                    threshold,
                )
                for threshold in sorted(set(args.threshold_sweep))
            },
        },
        "metadata_derivation": {"occlusion_level": "truncated > occluded > visible from manifest booleans", "vehicle_size": "crop width*height: small<12288, medium<49152, large otherwise", "lighting": "night flag else gray_mean: <64 low_light, <128 moderate_light, otherwise daylight", "label_confidence": "approved explicit label=high; approved unknown=medium; otherwise unknown"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "split": args.split, "rows": len(rows), "body_type": report["body_type"], "color": report["color"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
