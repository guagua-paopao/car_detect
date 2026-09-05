#!/usr/bin/env python3
"""Validation-selected probability fusion for two body-type checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attribute_dataset import VehicleAttributeDataset
from src.common import load_json
from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint


def metrics(pred, target, conf, labels, threshold):
    unknown = labels.index("unknown")
    valid = [i for i, t in enumerate(target) if t != -100]
    selected = [i for i in valid if pred[i] != unknown and conf[i] >= threshold]
    return {
        "evaluated": len(valid),
        "accuracy": sum(pred[i] == target[i] for i in valid) / len(valid) if valid else 0.0,
        "high_confidence_precision": sum(pred[i] == target[i] for i in selected) / len(selected) if selected else 0.0,
        "high_confidence_coverage": len(selected) / len(valid) if valid else 0.0,
        "high_confidence_selected": len(selected),
        "predicted_unknown_rate": sum(pred[i] == unknown for i in valid) / len(valid) if valid else 0.0,
        "true_unknown_rate": sum(target[i] == unknown for i in valid) / len(valid) if valid else 0.0,
    }


def logits_for(args, split, models, temps, labels, transform, device):
    import torch
    from torch.utils.data import DataLoader
    ds = VehicleAttributeDataset(args.manifest, split=split, body_types=labels, colors=["unknown"], transform=transform, training=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    outputs = [[] for _ in models]; targets = []
    for model in models: model.eval()
    with torch.no_grad():
        for images, body_target, _, _ in loader:
            images = images.to(device)
            for idx, model in enumerate(models):
                body_logits, _ = model(images)
                outputs[idx].append((body_logits / temps[idx]).cpu())
            targets.extend(body_target.tolist())
    return [torch.cat(chunks, dim=0) for chunks in outputs], targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--candidate-checkpoint", type=Path, required=True)
    ap.add_argument("--baseline-checkpoint", type=Path, required=True)
    ap.add_argument("--candidate-calibration", type=Path, required=True)
    ap.add_argument("--baseline-calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from torchvision import transforms

    label_data = load_json(args.labels)
    body_labels = label_data["body_types"]
    candidate_cal = json.loads(args.candidate_calibration.read_text(encoding="utf-8"))
    baseline_cal = json.loads(args.baseline_calibration.read_text(encoding="utf-8"))
    candidate_temp = float(candidate_cal["body_type"]["temperature"])
    baseline_temp = float(baseline_cal["body_type"]["temperature"])
    candidate_ck = torch.load(args.candidate_checkpoint, map_location="cpu")
    baseline_ck = torch.load(args.baseline_checkpoint, map_location="cpu")
    candidate_model = model_from_checkpoint(candidate_ck, pretrained=False)
    baseline_model = model_from_checkpoint(baseline_ck, pretrained=False)
    candidate_model.load_state_dict(candidate_ck["model_state"])
    baseline_model.load_state_dict(baseline_ck["model_state"])
    device = torch.device(args.device)
    candidate_model.to(device); baseline_model.to(device)
    size = int(candidate_ck.get("input_size", 256))
    transform = transforms.Compose([transforms.Resize((size, size), antialias=True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_logits, val_target = logits_for(args, "validation", [candidate_model, baseline_model], [candidate_temp, baseline_temp], body_labels, transform, device)
    test_logits, test_target = logits_for(args, "test", [candidate_model, baseline_model], [candidate_temp, baseline_temp], body_labels, transform, device)
    trials = []
    for alpha in [round(x / 20, 2) for x in range(0, 21)]:
        val_prob = alpha * val_logits[0].softmax(1) + (1.0 - alpha) * val_logits[1].softmax(1)
        conf, pred = val_prob.max(1)
        m = metrics(pred.tolist(), val_target, conf.tolist(), body_labels, args.threshold)
        trials.append({"alpha_candidate": alpha, **m})
    valid = [trial for trial in trials if trial["high_confidence_precision"] >= 0.93 and trial["high_confidence_coverage"] >= 0.45]
    if not valid:
        alpha = 1.0
    else:
        alpha = max(valid, key=lambda trial: (trial["high_confidence_coverage"], trial["high_confidence_precision"]))["alpha_candidate"]
    def evaluate(logits, target):
        prob = alpha * logits[0].softmax(1) + (1.0 - alpha) * logits[1].softmax(1)
        conf, pred = prob.max(1)
        return metrics(pred.tolist(), target, conf.tolist(), body_labels, args.threshold)
    report = {
        "schema_version": "body-probability-ensemble-v1",
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "candidate_temperature": candidate_temp,
        "baseline_temperature": baseline_temp,
        "selection": {"selected_alpha_candidate": alpha, "threshold": args.threshold, "validation_trials": trials},
        "validation": evaluate(val_logits, val_target),
        "test": evaluate(test_logits, test_target),
        "frozen_video_used": False,
    }
    report["release_body_target_met"] = all(report[part]["high_confidence_precision"] >= .93 and report[part]["high_confidence_coverage"] >= .45 for part in ("validation", "test"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
