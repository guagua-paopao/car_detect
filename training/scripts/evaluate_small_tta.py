#!/usr/bin/env python3
"""Validation-selected stretch/letterbox-like TTA audit for small body types."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attribute_dataset import VehicleAttributeDataset
from src.common import load_json
from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint

def collect(args, split, model, transforms, labels, device):
    import torch
    from torch.utils.data import DataLoader
    outputs = []
    targets = None
    sizes = None
    for transform in transforms:
        ds = VehicleAttributeDataset(args.manifest, split=split, body_types=labels, colors=["unknown"], transform=transform, training=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
        chunks = []
        with torch.no_grad():
            for images, target, _, _ in loader:
                logits, _ = model(images.to(device))
                chunks.append(logits.cpu())
        outputs.append(torch.cat(chunks, dim=0))
        if targets is None:
            targets = target.new_tensor([x for _, x, _, _ in []]) if False else []
            # Dataset order is deterministic; use a separate cheap target pass below.
            targets = [int(row["body_type_id"]) if "body_type_id" in row else 0 for row in []]
            targets = []
            for _, body_target, _, _ in DataLoader(ds, batch_size=256, shuffle=False, num_workers=0):
                targets.extend(body_target.tolist())
            sizes = [str(row.get("vehicle_size", "unknown")) for row in ds.rows]
    return outputs, targets, sizes

def summarize(prob, targets, sizes, unknown, threshold):
    conf, pred = prob.max(1)
    valid = [i for i, target in enumerate(targets) if target != -100]
    selected = [i for i in valid if int(pred[i]) != unknown and float(conf[i]) >= threshold]
    small = [i for i in valid if sizes[i] == "small"]
    small_selected = [i for i in small if int(pred[i]) != unknown and float(conf[i]) >= threshold]
    def m(indices, chosen):
        return {"evaluated": len(indices), "precision": sum(int(pred[i]) == targets[i] for i in chosen) / len(chosen) if chosen else 0.0, "coverage": len(chosen) / len(indices) if indices else 0.0, "selected": len(chosen)}
    return {"overall": m(valid, selected), "small": m(small, small_selected), "accuracy": sum(int(pred[i]) == targets[i] for i in valid) / len(valid) if valid else 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from torchvision import transforms
    labels = load_json(args.labels)["body_types"]
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = model_from_checkpoint(ck, pretrained=False); model.load_state_dict(ck["model_state"])
    device = torch.device(args.device); model.to(device).eval(); size = int(ck.get("input_size", 256))
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    stretch = transforms.Compose([transforms.Resize((size, size), antialias=True), *norm])
    preserve = transforms.Compose([transforms.Resize(size, antialias=True), transforms.CenterCrop(size), *norm])
    cal = json.loads(args.calibration.read_text(encoding="utf-8")); temperature = float(cal["body_type"]["temperature"])
    val_out, val_t, val_s = collect(args, "validation", model, [stretch, preserve], labels, device)
    test_out, test_t, test_s = collect(args, "test", model, [stretch, preserve], labels, device)
    trials = []
    for alpha in [round(x / 20, 2) for x in range(0, 21)]:
        for threshold in [round(x / 100, 2) for x in range(45, 91, 5)]:
            prob = (alpha * (val_out[1] / temperature).softmax(1) + (1-alpha) * (val_out[0] / temperature).softmax(1))
            trials.append({"alpha_preserve": alpha, "threshold": threshold, **summarize(prob, val_t, val_s, labels.index("unknown"), threshold)})
    valid = [x for x in trials if x["overall"]["precision"] >= .93 and x["overall"]["coverage"] >= .45 and x["small"]["precision"] >= .93]
    selected = max(valid, key=lambda x: (x["small"]["coverage"], x["overall"]["coverage"], x["overall"]["precision"])) if valid else {"alpha_preserve": 0.0, "threshold": 0.75}
    alpha, threshold = float(selected["alpha_preserve"]), float(selected["threshold"])
    def evaluate(out, target, sizes):
        prob = alpha * (out[1] / temperature).softmax(1) + (1-alpha) * (out[0] / temperature).softmax(1)
        return summarize(prob, target, sizes, labels.index("unknown"), threshold)
    report = {"schema_version": "small-tta-evaluation-v1", "checkpoint": str(args.checkpoint), "temperature": temperature, "selection": {"alpha_preserve": alpha, "threshold": threshold, "validation_trials": trials}, "validation": evaluate(val_out, val_t, val_s), "test": evaluate(test_out, test_t, test_s), "frozen_video_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
