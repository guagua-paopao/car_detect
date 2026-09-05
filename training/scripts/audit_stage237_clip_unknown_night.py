#!/usr/bin/env python3
"""Semantic re-audit of unknown color-training scenes with OpenCLIP.

The model is calibrated only on train-pool agent decisions.  It produces
night candidates for later visual review; it never changes color or scene
truth by itself and never opens validation, test, or frozen-video pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROMPTS = {
    "night": [
        "a traffic surveillance crop of a vehicle outdoors at night",
        "a vehicle driving at night under street lights",
        "a nighttime CCTV image of a vehicle illuminated by headlights",
    ],
    "low_light": [
        "a traffic camera image of a vehicle at dusk or in low light",
        "a vehicle outdoors in a dim evening scene",
    ],
    "daylight": [
        "a traffic surveillance crop of a vehicle outdoors during daylight",
        "a vehicle driving in daytime sunlight",
    ],
    "catalog": [
        "an online car listing or dealership photograph of a vehicle",
        "a studio product photograph of a vehicle",
    ],
    "daylight_dark_vehicle": [
        "a black or dark colored vehicle photographed during daylight",
        "a dark vehicle in a bright daytime scene",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def choose_threshold(scores: list[float], labels: list[int], minimum_precision: float) -> dict:
    positives = sum(labels)
    best = None
    for threshold in sorted(set(scores), reverse=True):
        predicted = [score >= threshold for score in scores]
        tp = sum(1 for flag, label in zip(predicted, labels) if flag and label == 1)
        fp = sum(1 for flag, label in zip(predicted, labels) if flag and label == 0)
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        candidate = {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }
        if precision >= minimum_precision and tp >= 8:
            key = (recall, precision, threshold)
            if best is None or key > best[0]:
                best = (key, candidate)
    return best[1] if best else {
        "threshold": None,
        "tp": 0,
        "fp": 0,
        "fn": positives,
        "precision": 0.0,
        "recall": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--calibration-overlay", type=Path, required=True)
    parser.add_argument("--nightowls-manifest", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-calibration", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-calibration-precision", type=float, default=0.90)
    parser.add_argument("--minimum-calibration-recall", type=float, default=0.40)
    parser.add_argument("--minimum-nightowls-recall", type=float, default=0.50)
    parser.add_argument("--maximum-known-daylight-fpr", type=float, default=0.05)
    parser.add_argument("--nightowls-diagnostic-sample", type=int, default=1200)
    parser.add_argument("--daylight-diagnostic-sample", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for output in (args.output_candidates, args.output_calibration, args.output_report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {output}")

    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    with args.active_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        active_rows = list(csv.DictReader(handle))
    with args.calibration_overlay.open("r", encoding="utf-8-sig", newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
    with args.nightowls_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        nightowls_rows = list(csv.DictReader(handle))

    calibration = []
    for row in calibration_rows:
        decision = row.get("visual_decision")
        scene = row.get("visual_scene")
        if decision == "accept" and scene in {"night", "low_light"}:
            label = 1
        elif decision == "reject" and scene == "unknown":
            label = 0
        else:
            continue
        calibration.append({"image_path": row["image_path"], "label": label, "source": row.get("source_dataset", "")})

    unknown_rows = [
        row for row in active_rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("stage177_effective_representative"))
        and row.get("stage177_scene_label") == "unknown"
        and not row.get("stage177_read_error")
    ]
    daylight_pool = [
        row for row in active_rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("stage177_effective_representative"))
        and row.get("stage177_scene_label") == "daylight"
        and not row.get("stage177_read_error")
    ]
    daylight_diagnostic = sorted(
        daylight_pool,
        key=lambda row: hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest(),
    )[: args.daylight_diagnostic_sample]
    nightowls_diagnostic = sorted(
        [row for row in nightowls_rows if truthy(row.get("crop_quality_usable"))],
        key=lambda row: hashlib.sha256(row["crop_path"].encode("utf-8")).hexdigest(),
    )[: args.nightowls_diagnostic_sample]

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        cache_dir=str(args.cache_dir),
        weights_only=False,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    device = torch.device(args.device)
    model.to(device).eval()
    class_names = list(PROMPTS)
    with torch.no_grad():
        prototypes = []
        for name in class_names:
            tokens = tokenizer(PROMPTS[name]).to(device)
            embeddings = model.encode_text(tokens, normalize=True)
            prototype = embeddings.mean(dim=0)
            prototypes.append(prototype / prototype.norm())
        text_features = torch.stack(prototypes)

    class ImageDataset(Dataset):
        def __init__(self, paths: list[str]):
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            path = Path(self.paths[index])
            try:
                with Image.open(path) as opened:
                    image = preprocess(opened.convert("RGB"))
                ok = True
            except Exception:
                image = torch.zeros(3, 224, 224)
                ok = False
            return image, index, ok

    def infer(paths: list[str]) -> list[dict[str, object]]:
        results: list[dict[str, object] | None] = [None] * len(paths)
        loader = DataLoader(
            ImageDataset(paths), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        with torch.no_grad():
            for images, indexes, readable in loader:
                image_features = model.encode_image(images.to(device, non_blocking=True), normalize=True)
                probabilities = (100.0 * image_features @ text_features.T).softmax(dim=1).cpu()
                for local, index in enumerate(indexes.tolist()):
                    probs = {name: float(probabilities[local, class_index]) for class_index, name in enumerate(class_names)}
                    adverse = probs["night"] + probs["low_light"]
                    results[index] = {
                        "readable": bool(readable[local]),
                        "adverse_score": adverse,
                        "probabilities": probs,
                        "predicted_class": max(probs, key=probs.get),
                    }
        return [result or {"readable": False, "adverse_score": 0.0, "probabilities": {}, "predicted_class": "error"} for result in results]

    calibration_paths = [row["image_path"] for row in calibration]
    calibration_predictions = infer(calibration_paths)
    readable_calibration = [
        (row, prediction) for row, prediction in zip(calibration, calibration_predictions)
        if prediction["readable"]
    ]
    scores = [float(prediction["adverse_score"]) for _, prediction in readable_calibration]
    labels = [int(row["label"]) for row, _ in readable_calibration]
    threshold_metrics = choose_threshold(scores, labels, args.minimum_calibration_precision)
    threshold = threshold_metrics["threshold"]

    calibration_output = []
    for row, prediction in zip(calibration, calibration_predictions):
        calibration_output.append({
            **row,
            "readable": str(prediction["readable"]).lower(),
            "adverse_score": f"{float(prediction['adverse_score']):.8f}",
            "predicted_class": prediction["predicted_class"],
            **{f"prob_{key}": f"{float(value):.8f}" for key, value in dict(prediction["probabilities"]).items()},
        })
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    with args.output_calibration.open("w", encoding="utf-8", newline="") as handle:
        fields = list(calibration_output[0]) if calibration_output else ["image_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(calibration_output)

    precheck_pass = (
        threshold is not None
        and threshold_metrics["precision"] >= args.minimum_calibration_precision
        and threshold_metrics["recall"] >= args.minimum_calibration_recall
        and auroc(scores, labels) >= 0.80
    )
    nightowls_recall = None
    daylight_fpr = None
    diagnostic_counts = Counter()
    if precheck_pass:
        night_predictions = infer([row["crop_path"] for row in nightowls_diagnostic])
        daylight_predictions = infer([row["image_path"] for row in daylight_diagnostic])
        night_flags = [prediction["readable"] and float(prediction["adverse_score"]) >= threshold for prediction in night_predictions]
        day_flags = [prediction["readable"] and float(prediction["adverse_score"]) >= threshold for prediction in daylight_predictions]
        nightowls_recall = sum(night_flags) / len(night_flags) if night_flags else 0.0
        daylight_fpr = sum(day_flags) / len(day_flags) if day_flags else 1.0
        for prediction in night_predictions:
            diagnostic_counts[f"nightowls_{prediction['predicted_class']}"] += 1
        for prediction in daylight_predictions:
            diagnostic_counts[f"known_daylight_{prediction['predicted_class']}"] += 1
        precheck_pass = (
            nightowls_recall >= args.minimum_nightowls_recall
            and daylight_fpr <= args.maximum_known_daylight_fpr
        )

    candidates = []
    candidate_sources = Counter()
    candidate_colors = Counter()
    if precheck_pass:
        unknown_predictions = infer([row["image_path"] for row in unknown_rows])
        for row, prediction in zip(unknown_rows, unknown_predictions):
            if not prediction["readable"] or float(prediction["adverse_score"]) < threshold:
                continue
            output = {
                "image_path": row["image_path"],
                "source_dataset": row.get("source_dataset", ""),
                "color": row.get("color", "unknown"),
                "stage177_group_key": row.get("stage177_group_key", ""),
                "stage177_scene_group_key": row.get("stage177_scene_group_key", ""),
                "track_key": row.get("track_key", ""),
                "camera_id": row.get("camera_id", ""),
                "video_id": row.get("video_id", ""),
                "vehicle_size": row.get("vehicle_size", ""),
                "stage177_dhash64": row.get("stage177_dhash64", ""),
                "adverse_score": f"{float(prediction['adverse_score']):.8f}",
                "predicted_class": prediction["predicted_class"],
                **{f"prob_{key}": f"{float(value):.8f}" for key, value in dict(prediction["probabilities"]).items()},
                "scene_proposal": "semantic_night_or_lowlight_candidate_not_truth",
                "training_eligible": "false",
            }
            candidates.append(output)
            candidate_sources[output["source_dataset"]] += 1
            candidate_colors[output["color"]] += 1

    fields = list(candidates[0]) if candidates else [
        "image_path", "source_dataset", "color", "adverse_score", "predicted_class",
        "scene_proposal", "training_eligible",
    ]
    with args.output_candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    model_files = sorted(path for path in args.cache_dir.rglob("*") if path.is_file())
    report = {
        "schema_version": "stage237-clip-unknown-night-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_candidates_pending_visual_audit" if precheck_pass else "complete_fail_closed_semantic_precheck",
        "model": args.model,
        "pretrained": args.pretrained,
        "trusted_torchscript_load": {
            "weights_only": False,
            "reason": "OpenAI official TorchScript archive verified against publisher SHA256 before load",
        },
        "model_files": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in model_files],
        "prompts": PROMPTS,
        "calibration_rows": len(calibration),
        "calibration_readable": len(readable_calibration),
        "calibration_positive": sum(labels),
        "calibration_negative": len(labels) - sum(labels),
        "calibration_auroc": auroc(scores, labels),
        "threshold_metrics": threshold_metrics,
        "nightowls_diagnostic_rows": len(nightowls_diagnostic) if threshold is not None else 0,
        "nightowls_recall": nightowls_recall,
        "known_daylight_diagnostic_rows": len(daylight_diagnostic) if threshold is not None else 0,
        "known_daylight_false_positive_rate": daylight_fpr,
        "diagnostic_predicted_class_counts": dict(sorted(diagnostic_counts.items())),
        "unknown_effective_rows_available": len(unknown_rows),
        "unknown_rows_scanned": len(unknown_rows) if precheck_pass else 0,
        "candidate_rows": len(candidates),
        "candidate_source_counts": dict(sorted(candidate_sources.items())),
        "candidate_color_counts": dict(sorted(candidate_colors.items())),
        "inputs": {
            "active_manifest": str(args.active_manifest.resolve()),
            "active_manifest_sha256": sha256(args.active_manifest),
            "calibration_overlay": str(args.calibration_overlay.resolve()),
            "calibration_overlay_sha256": sha256(args.calibration_overlay),
            "nightowls_manifest": str(args.nightowls_manifest.resolve()),
            "nightowls_manifest_sha256": sha256(args.nightowls_manifest),
        },
        "outputs": {
            "candidates": str(args.output_candidates.resolve()),
            "candidates_sha256": sha256(args.output_candidates),
            "calibration": str(args.output_calibration.resolve()),
            "calibration_sha256": sha256(args.output_calibration),
        },
        "policy": {
            "semantic_predictions_are_candidates_not_truth": True,
            "no_color_or_scene_labels_modified": True,
            "visual_audit_required_before_recount": True,
            "train_pixels_only": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if precheck_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
