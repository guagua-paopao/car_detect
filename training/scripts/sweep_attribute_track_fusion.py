#!/usr/bin/env python3
"""Validation-only sweep of per-head thresholds and quality-weighted track fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: dict) -> dict:
    keys = ("evaluated", "selected", "precision", "coverage", "effective_unknown_rate", "correct_over_all")
    return {key: value.get(key) for key in keys}


def compact_stability(value: dict) -> dict:
    keys = ("windows", "emitted_predictions", "weighted_stability_rate", "windows_stability_ge_0_95_rate", "label_switches_total", "abstention_rate")
    return {key: value.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--minimum-window-frames", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--minimum-shares", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--minimum-margins", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    training_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(training_root))
    from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint
    from evaluate_attribute_track_fusion import frame_metrics, fuse_sequence, quality_weight, stability_metrics

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = [row for row in csv.DictReader(handle) if row.get("split") == args.split]
    window_counts = Counter(row.get("window_id", "") for row in all_rows)
    rows = [
        row for row in all_rows
        if row.get("window_id", "") and window_counts[row["window_id"]] >= args.minimum_window_frames
    ]
    rows.sort(key=lambda row: (row["window_id"], int(row.get("window_pos", 0))))
    if not rows:
        raise RuntimeError("no repeated-track rows")
    windows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        windows[row["window_id"]].append(index)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    body_labels = [str(value) for value in checkpoint["body_types"]]
    color_labels = [str(value) for value in checkpoint["colors"]]
    input_size = int(checkpoint.get("input_size", 224))
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class Rows(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(args.manifest.parent / rows[index]["image_path"]) as image:
                return transform(image.convert("RGB")), index

    loader = DataLoader(Rows(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    raw = [None] * len(rows)
    with torch.no_grad():
        for images, indexes in loader:
            body_logits, color_logits = model(images.to(device, non_blocking=True))
            body_conf, body_index = body_logits.softmax(1).max(1)
            color_conf, color_index = color_logits.softmax(1).max(1)
            for position, index in enumerate(indexes.tolist()):
                raw[index] = {
                    "body": body_labels[int(body_index[position])],
                    "body_confidence": float(body_conf[position]),
                    "color": color_labels[int(color_index[position])],
                    "color_confidence": float(color_conf[position]),
                    "quality": quality_weight(rows[index]),
                }
    if any(value is None for value in raw):
        raise RuntimeError("inference incomplete")

    report_heads = {}
    for head, truth_key in (("body_type", "track_body_truth"), ("color", "track_color_truth")):
        prediction_key = "body" if head == "body_type" else "color"
        confidence_key = f"{prediction_key}_confidence"
        truths = [row.get(truth_key, "unknown") or "unknown" for row in rows]
        grid = []
        for threshold in sorted(set(args.thresholds)):
            single = [
                item[prediction_key]
                if item[prediction_key] != "unknown" and item[confidence_key] >= threshold
                else "unknown"
                for item in raw
            ]
            single_metrics = frame_metrics(single, truths)
            for window_size in sorted(set(args.windows)):
                for share in sorted(set(args.minimum_shares)):
                    for margin in sorted(set(args.minimum_margins)):
                        fused = ["unknown"] * len(rows)
                        for indexes in windows.values():
                            values = [
                                {
                                    "label": single[index],
                                    "confidence": raw[index][confidence_key],
                                    "quality": raw[index]["quality"],
                                }
                                for index in indexes
                            ]
                            outputs = fuse_sequence(values, window_size, share, margin)
                            for index, label in zip(indexes, outputs):
                                fused[index] = label
                        final_predictions = [fused[indexes[-1]] for indexes in windows.values()]
                        final_truths = [truths[indexes[-1]] for indexes in windows.values()]
                        final = frame_metrics(final_predictions, final_truths)
                        stability = stability_metrics({
                            key: [fused[index] for index in indexes] for key, indexes in windows.items()
                        })
                        grid.append({
                            "threshold": threshold, "window": window_size,
                            "minimum_share": share, "minimum_margin": margin,
                            "single": compact(single_metrics),
                            "final_window": compact(final),
                            "stability": compact_stability(stability),
                        })
        eligible = [
            item for item in grid
            if (item["final_window"]["selected"] or 0) > 0
            and (item["final_window"]["precision"] or 0.0) >= 0.93
            and (item["stability"]["weighted_stability_rate"] or 0.0) >= 0.95
        ]
        if eligible:
            selected = max(eligible, key=lambda item: (
                item["final_window"]["coverage"] or 0.0,
                item["single"]["coverage"] or 0.0,
                -(item["stability"]["abstention_rate"] or 1.0),
                -item["threshold"],
            ))
            selection_status = "precision_and_stability_gate_found"
        else:
            selected = max(grid, key=lambda item: (
                item["final_window"]["precision"] or 0.0,
                item["final_window"]["coverage"] or 0.0,
                item["stability"]["weighted_stability_rate"] or 0.0,
            ))
            selection_status = "no_precision_stability_gate_configuration"
        selected_predictions = [
            item[prediction_key]
            if item[prediction_key] != "unknown" and item[confidence_key] >= selected["threshold"]
            else "unknown"
            for item in raw
        ]
        selected_fused = ["unknown"] * len(rows)
        for indexes in windows.values():
            values = [
                {"label": selected_predictions[index], "confidence": raw[index][confidence_key], "quality": raw[index]["quality"]}
                for index in indexes
            ]
            outputs = fuse_sequence(values, selected["window"], selected["minimum_share"], selected["minimum_margin"])
            for index, label in zip(indexes, outputs):
                selected_fused[index] = label
        strata = {}
        for lighting in sorted({row.get("lighting", "unknown") or "unknown" for row in rows}):
            indexes = [i for i, row in enumerate(rows) if (row.get("lighting", "unknown") or "unknown") == lighting]
            strata[lighting] = compact(frame_metrics(
                [selected_fused[index] for index in indexes], [truths[index] for index in indexes]
            ))
        report_heads[head] = {
            "selection_status": selection_status,
            "selected": selected,
            "selected_frame_stratified_by_lighting_proxy": strata,
            "grid": grid,
        }

    report = {
        "schema_version": "attribute-track-fusion-sweep-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "validation", "test_used": False, "frozen_video_used": False,
            "minimum_window_frames": args.minimum_window_frames,
            "selection": "maximum final-window coverage subject to precision >= 0.93 and weighted stability >= 0.95",
            "lighting": "pixel-derived proxy, not source ground truth",
            "license": "inherited from manifest; VFG-7 is CC BY-NC 4.0 evaluation-only",
        },
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "rows": len(rows), "windows": len(windows),
        "body_type": report_heads["body_type"], "color": report_heads["color"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "output_sha256": sha256(args.output),
        "rows": len(rows), "windows": len(windows),
        "body_type": {key: value for key, value in report_heads["body_type"].items() if key != "grid"},
        "color": {key: value for key, value in report_heads["color"].items() if key != "grid"},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
