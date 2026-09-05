#!/usr/bin/env python3
"""Evaluate single-frame and quality-weighted track fusion on true track IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_weight(row: dict[str, str]) -> float:
    try:
        area = max(1.0, float(row.get("width", 0)) * float(row.get("height", 0)))
    except (TypeError, ValueError):
        area = 1.0
    try:
        edge = max(0.0, float(row.get("edge_variance", 0)))
    except (TypeError, ValueError):
        edge = 0.0
    size_quality = max(0.10, min(1.0, math.sqrt(area / (256.0 * 192.0))))
    detail_quality = max(0.10, min(1.0, math.log1p(edge) / math.log1p(1000.0)))
    quality = size_quality * detail_quality
    if parse_bool(row.get("blur")):
        quality *= 0.60
    if parse_bool(row.get("occluded")):
        quality *= 0.45
    if parse_bool(row.get("truncated")):
        quality *= 0.55
    if str(row.get("crop_quality", "")).strip().lower() == "usable":
        quality *= 0.75
    return max(0.01, min(1.0, quality))


def fuse_sequence(
    values: list[dict],
    window: int,
    minimum_share: float,
    minimum_margin: float,
) -> list[str]:
    output: list[str] = []
    for index in range(len(values)):
        history = values[max(0, index - window + 1):index + 1]
        scores: defaultdict[str, float] = defaultdict(float)
        for value in history:
            if value["label"] == "unknown":
                continue
            scores[value["label"]] += value["confidence"] * value["quality"]
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        total = sum(scores.values())
        if not ranked or total <= 0:
            output.append("unknown")
            continue
        share = ranked[0][1] / total
        margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
        output.append(ranked[0][0] if share >= minimum_share and margin >= minimum_margin else "unknown")
    return output


def frame_metrics(predictions: list[str], truths: list[str]) -> dict:
    valid = [index for index, truth in enumerate(truths) if truth not in {"", "unknown"}]
    selected = [index for index in valid if predictions[index] != "unknown"]
    correct = sum(predictions[index] == truths[index] for index in selected)
    return {
        "evaluated": len(valid),
        "selected": len(selected),
        "precision": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(valid) if valid else 0.0,
        "effective_unknown_rate": 1.0 - len(selected) / len(valid) if valid else 0.0,
        "correct_over_all": correct / len(valid) if valid else 0.0,
        "predicted_counts": dict(sorted(Counter(predictions[index] for index in valid).items())),
        "truth_counts": dict(sorted(Counter(truths[index] for index in valid).items())),
        "confusion_counts": dict(sorted(Counter(
            f"{truths[index]}->{predictions[index]}" for index in valid
        ).items())),
    }


def stability_metrics(groups: dict[str, list[str]]) -> dict:
    tracks = 0
    emitted = 0
    dominant = 0
    switches = 0
    stable_tracks = 0
    abstained = 0
    total = 0
    per_track = []
    for key, sequence in sorted(groups.items()):
        if len(sequence) < 3:
            continue
        tracks += 1
        total += len(sequence)
        abstained += sum(label == "unknown" for label in sequence)
        labels = [label for label in sequence if label != "unknown"]
        local_switches = sum(left != right for left, right in zip(labels, labels[1:]))
        local_dominant = max(Counter(labels).values()) if labels else 0
        local_stability = local_dominant / len(labels) if labels else 0.0
        emitted += len(labels)
        dominant += local_dominant
        switches += local_switches
        stable_tracks += int(local_stability >= 0.95)
        per_track.append({
            "window_id": key,
            "frames": len(sequence),
            "emitted": len(labels),
            "stability": local_stability,
            "switches": local_switches,
            "abstention_rate": 1.0 - len(labels) / len(sequence),
        })
    return {
        "windows": tracks,
        "emitted_predictions": emitted,
        "weighted_stability_rate": dominant / emitted if emitted else 0.0,
        "windows_stability_ge_0_95_rate": stable_tracks / tracks if tracks else 0.0,
        "label_switches_total": switches,
        "abstention_rate": abstained / total if total else 0.0,
        "per_window": per_track,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--type-threshold", type=float, required=True)
    parser.add_argument("--color-threshold", type=float, required=True)
    parser.add_argument("--body-temperature", type=float, default=1.0)
    parser.add_argument("--color-temperature", type=float, default=1.0)
    parser.add_argument("--fusion-windows", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--fusion-min-share", type=float, default=0.60)
    parser.add_argument("--fusion-min-margin", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--minimum-window-frames", type=int, default=1,
        help="Discard short/non-track windows before inference (use 3 for repeated-track-only evaluation).",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    training_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(training_root))
    from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == args.split]
    rows.sort(key=lambda row: (row.get("window_id", ""), int(row.get("window_pos", 0))))
    if args.minimum_window_frames > 1:
        window_counts = Counter(row.get("window_id", "") for row in rows)
        rows = [
            row for row in rows
            if row.get("window_id", "") and window_counts[row.get("window_id", "")] >= args.minimum_window_frames
        ]
    if not rows:
        raise RuntimeError(f"no rows for split {args.split}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    body_labels = [str(value) for value in checkpoint["body_types"]]
    color_labels = [str(value) for value in checkpoint["colors"]]
    input_size = int(checkpoint.get("input_size", 224))
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class Rows(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            with Image.open(args.manifest.parent / rows[index]["image_path"]) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, index

    loader = DataLoader(Rows(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions: list[dict | None] = [None] * len(rows)
    with torch.no_grad():
        for images, indexes in loader:
            body_logits, color_logits = model(images.to(device, non_blocking=True))
            body_prob = (body_logits / args.body_temperature).softmax(1)
            color_prob = (color_logits / args.color_temperature).softmax(1)
            body_conf, body_index = body_prob.max(1)
            color_conf, color_index = color_prob.max(1)
            for position, index in enumerate(indexes.tolist()):
                body_label = body_labels[int(body_index[position])]
                color_label = color_labels[int(color_index[position])]
                body_score = float(body_conf[position])
                color_score = float(color_conf[position])
                predictions[index] = {
                    "body": body_label if body_label != "unknown" and body_score >= args.type_threshold else "unknown",
                    "body_confidence": body_score,
                    "color": color_label if color_label != "unknown" and color_score >= args.color_threshold else "unknown",
                    "color_confidence": color_score,
                    "quality": quality_weight(rows[index]),
                }
    if any(value is None for value in predictions):
        raise RuntimeError("inference did not produce all predictions")
    typed_predictions = [value for value in predictions if value is not None]

    windows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        windows[row["window_id"]].append(index)
    for indexes in windows.values():
        indexes.sort(key=lambda index: int(rows[index].get("window_pos", 0)))

    report_heads = {}
    strata_fields = ("weather", "vehicle_size", "occlusion_level")

    def stratified_stability(labels: list[str], field: str) -> dict:
        result = {}
        for value in sorted({row.get(field, "unknown") or "unknown" for row in rows}):
            groups = {}
            for key, indexes in windows.items():
                selected = [index for index in indexes if (rows[index].get(field, "unknown") or "unknown") == value]
                if selected:
                    groups[key] = [labels[index] for index in selected]
            metrics = stability_metrics(groups)
            metrics.pop("per_window", None)
            result[value] = metrics
        return result

    for head, truth_key in (("body_type", "track_body_truth"), ("color", "track_color_truth")):
        prediction_key = "body" if head == "body_type" else "color"
        confidence_key = f"{prediction_key}_confidence"
        truths = [row.get(truth_key, "unknown") for row in rows]
        single = [value[prediction_key] for value in typed_predictions]
        head_report = {
            "single_frame": frame_metrics(single, truths),
            "single_frame_stability": stability_metrics({
                key: [single[index] for index in indexes] for key, indexes in windows.items()
            }),
            "single_frame_stability_stratified": {
                field: stratified_stability(single, field) for field in strata_fields
            },
            "fusion": {},
        }
        for window_size in sorted(set(args.fusion_windows)):
            fused = ["unknown"] * len(rows)
            for key, indexes in windows.items():
                values = [
                    {
                        "label": typed_predictions[index][prediction_key],
                        "confidence": typed_predictions[index][confidence_key],
                        "quality": typed_predictions[index]["quality"],
                    }
                    for index in indexes
                ]
                labels = fuse_sequence(values, window_size, args.fusion_min_share, args.fusion_min_margin)
                for index, label in zip(indexes, labels):
                    fused[index] = label
            final_predictions = []
            final_truths = []
            for indexes in windows.values():
                final_predictions.append(fused[indexes[-1]])
                final_truths.append(truths[indexes[-1]])
            head_report["fusion"][str(window_size)] = {
                "frame": frame_metrics(fused, truths),
                "stability": stability_metrics({
                    key: [fused[index] for index in indexes] for key, indexes in windows.items()
                }),
                "stability_stratified": {
                    field: stratified_stability(fused, field) for field in strata_fields
                },
                "final_window": frame_metrics(final_predictions, final_truths),
            }
        strata = {}
        for field in strata_fields:
            values = sorted({row.get(field, "unknown") or "unknown" for row in rows})
            strata[field] = {}
            for value in values:
                indexes = [index for index, row in enumerate(rows) if (row.get(field, "unknown") or "unknown") == value]
                strata[field][value] = frame_metrics([single[index] for index in indexes], [truths[index] for index in indexes])
        head_report["single_frame_stratified"] = strata
        report_heads[head] = head_report

    report = {
        "schema_version": "uadetrac-track-fusion-evaluation-v1",
        "protocol": {
            "split": args.split,
            "test_used_for_selection": False,
            "frozen_video_used": False,
            "official_track_ids_used": True,
            "quality_weight": "area*edge detail with blur/occlusion/truncation/usable penalties",
            "fusion_windows": sorted(set(args.fusion_windows)),
            "fusion_min_share": args.fusion_min_share,
            "fusion_min_margin": args.fusion_min_margin,
            "minimum_window_frames": args.minimum_window_frames,
            "thresholds": {"body_type": args.type_threshold, "color": args.color_threshold},
            "temperatures": {"body_type": args.body_temperature, "color": args.color_temperature},
        },
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "rows": len(rows),
        "windows": len(windows),
        "body_type": report_heads["body_type"],
        "color": report_heads["color"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def compact_stability(value: dict) -> dict:
        return {key: item for key, item in value.items() if key != "per_window"}
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "windows": len(windows),
        "body_type": {
            "single": report["body_type"]["single_frame"],
            "single_stability": compact_stability(report["body_type"]["single_frame_stability"]),
            "fusion": {key: {
                "frame": value["frame"],
                "stability": compact_stability(value["stability"]),
                "final_window": value["final_window"],
            } for key, value in report["body_type"]["fusion"].items()},
        },
        "color": {
            "single": report["color"]["single_frame"],
            "single_stability": compact_stability(report["color"]["single_frame_stability"]),
            "fusion": {key: {
                "frame": value["frame"],
                "stability": compact_stability(value["stability"]),
                "final_window": value["final_window"],
            } for key, value in report["color"]["fusion"].items()},
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
