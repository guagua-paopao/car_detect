#!/usr/bin/env python3
"""Analyze train-only BMD raw color labels with track-level logit aggregation.

This is an analysis gate only: it never reads validation/test rows and never
writes a training manifest. It reports how many tracks would survive a
conservative aggregate-consensus policy before any pseudo-label training.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.multitask_mobilenet_v3 import model_from_checkpoint, IMAGENET_MEAN, IMAGENET_STD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate-temperature", type=float, default=.825)
    ap.add_argument("--baseline-temperature", type=float, default=.75)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    with a.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    targets = [(i, r) for i, r in enumerate(rows)
               if r.get("split") == "train"
               and r.get("source_dataset") in {"BMD-45-RAW", "BMD-45-RAW-COCO"}
               and r.get("color_supervised", "false").lower() in {"false", "0", "no"}]
    if not targets:
        raise RuntimeError("no unlabeled BMD raw train rows")

    cck = torch.load(a.candidate, map_location="cpu")
    bck = torch.load(a.baseline, map_location="cpu")
    cm = model_from_checkpoint(cck, pretrained=False)
    bm = model_from_checkpoint(bck, pretrained=False)
    cm.load_state_dict(cck["model_state"]); bm.load_state_dict(bck["model_state"])
    cm.eval(); bm.eval()
    dev = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    cm.to(dev); bm.to(dev)
    colors = list(cck["colors"]); unknown = colors.index("unknown")
    size = int(cck.get("input_size", 224))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize = transforms.Resize((size, size), antialias=True)
    to_tensor = transforms.ToTensor()

    class D(Dataset):
        def __len__(self): return len(targets)
        def __getitem__(self, j):
            idx, r = targets[j]
            with Image.open((a.manifest.parent / r["image_path"]).resolve()) as im:
                im = im.convert("RGB")
            xs = []
            for bright, contrast in [(1., 1.), (.78, 1.08), (1.22, .92)]:
                x = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(bright)).enhance(contrast)
                xs.append(norm(to_tensor(resize(x))))
            return torch.stack(xs), j

    loader = DataLoader(D(), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=dev.type == "cuda")
    cp = torch.zeros((len(targets), len(colors)))
    bp = torch.zeros_like(cp)
    with torch.no_grad():
        for batch, js in loader:
            x = batch.to(dev)
            c_aug, b_aug = [], []
            for k in range(3):
                _, cl = cm(x[:, k]); _, bl = bm(x[:, k])
                c_aug.append((cl / a.candidate_temperature).softmax(1))
                b_aug.append((bl / a.baseline_temperature).softmax(1))
            cmean = torch.stack(c_aug).mean(0).cpu()
            bmean = torch.stack(b_aug).mean(0).cpu()
            cp[js] = cmean; bp[js] = bmean

    groups = defaultdict(list)
    for j, (_, r) in enumerate(targets):
        groups[(r.get("video_id", ""), r.get("track_group", ""))].append(j)

    def summarize(conf_gate: float, agree_gate: float, min_frames: int) -> dict:
        accepted = []
        for key, js in groups.items():
            if len(js) < min_frames: continue
            cmean = cp[js].mean(0); bmean = bp[js].mean(0)
            cnon = cmean.clone(); bnon = bmean.clone()
            cnon[unknown] = -1; bnon[unknown] = -1
            cc, cl = cnon.max(0); bc, bl = bnon.max(0)
            # Track-level agreement is measured against the aggregate label.
            ca = float((cp[js].argmax(1) == cl).float().mean())
            ba = float((bp[js].argmax(1) == bl).float().mean())
            same = cl == bl
            margin_c = float((torch.topk(cnon, 2).values[0] - torch.topk(cnon, 2).values[1]))
            margin_b = float((torch.topk(bnon, 2).values[0] - torch.topk(bnon, 2).values[1]))
            if same and int(cl) != unknown and float(cc) >= conf_gate and float(bc) >= conf_gate and ca >= agree_gate and ba >= agree_gate:
                accepted.extend((j, int(cl)) for j in js)
        counts = Counter(colors[p] for _, p in accepted)
        return {"confidence_gate": conf_gate, "agreement_gate": agree_gate,
                "min_track_frames": min_frames, "accepted_rows": len(accepted),
                "accepted_tracks": len({(targets[j][1].get('video_id',''), targets[j][1].get('track_group','')) for j, _ in accepted}),
                "accepted_color_counts": dict(counts)}

    out = {
        "schema_version": "bmd-raw-color-track-logit-analysis-v1",
        "manifest": str(a.manifest), "rows": len(rows), "train_targets": len(targets),
        "groups": len(groups), "group_size_histogram": dict(Counter(map(len, groups.values()))),
        "candidate": str(a.candidate), "baseline": str(a.baseline),
        "temperatures": {"candidate": a.candidate_temperature, "baseline": a.baseline_temperature},
        "policy": "train-only BMD raw; three photometric variants averaged per frame, then logits/probabilities aggregated per track; analysis only",
        "frozen_video_used": False, "validation_or_test_used": False,
        "grid": [summarize(c, g, m) for c in (.5, .6, .7, .8, .9) for g in (.5, .7, .8, .9) for m in (3, 5)],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_targets": len(targets), "groups": len(groups), "grid": out["grid"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
