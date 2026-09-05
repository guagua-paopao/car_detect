#!/usr/bin/env python3
"""Fail-closed teacher/augmentation consensus audit for BMD small body labels."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_small(row: dict[str, str]) -> bool:
    return str(row.get("small_target", "")).strip().lower() in {"1", "true", "yes"} or row.get("vehicle_size", "").strip().lower() == "small"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--resnet", type=Path, required=True)
    ap.add_argument("--convnext", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confidence", type=float, default=0.60)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    fields = list(rows[0])
    target_indices = [i for i, row in enumerate(rows) if row.get("split") == "train" and row.get("source_dataset") == "BMD-45" and is_small(row) and row.get("body_type") not in {"", "unknown"}]
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["body_types"]
    class RowDataset(Dataset):
        def __init__(self, selected, transform): self.selected, self.transform = selected, transform
        def __len__(self): return len(self.selected)
        def __getitem__(self, index):
            row = rows[self.selected[index]]
            with Image.open(args.manifest.parent / row["image_path"]) as image: image = image.convert("RGB")
            return self.transform(image), index
    checkpoints = [("baseline", args.baseline), ("resnet", args.resnet), ("convnext", args.convnext)]
    device = torch.device(args.device)
    models = []
    max_size = 256
    for name, path in checkpoints:
        ck = torch.load(path, map_location="cpu")
        max_size = max(max_size, int(ck.get("input_size", 256)))
        model = model_from_checkpoint(ck, pretrained=False); model.load_state_dict(ck["model_state"]); model.to(device).eval(); models.append((name, model, ck))
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    views = [transforms.Compose([transforms.Resize((max_size, max_size), antialias=True), *norm]), transforms.Compose([transforms.Resize(max_size, antialias=True), transforms.CenterCrop(max_size), *norm])]
    predictions = {name: [[None, None] for _ in target_indices] for name, _, _ in models}
    for view_index, transform in enumerate(views):
        loader = DataLoader(RowDataset(target_indices, transform), batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
        with torch.no_grad():
            for images, local_indices in loader:
                images = images.to(device)
                for name, model, _ in models:
                    logits, _ = model(images); probs = logits.softmax(1); conf, pred = probs.max(1)
                    for pred_value, conf_value, local in zip(pred.cpu().tolist(), conf.cpu().tolist(), local_indices.tolist()):
                        predictions[name][local][view_index] = (int(pred_value), float(conf_value))
    unknown = labels.index("unknown")
    accepted = 0; disagreement = 0; aug_conflict = 0; teacher_conflict = 0; original_mismatch = 0; accepted_classes = Counter(); rejected_classes = Counter()
    for local, row_index in enumerate(target_indices):
        row = rows[row_index]
        all_views = [predictions[name][local] for name, _, _ in models]
        view_preds = [p for pair in all_views for p, _ in pair]
        view_confs = [c for pair in all_views for _, c in pair]
        same_aug = all(pair[0][0] == pair[1][0] for pair in all_views)
        same_teacher = len({pair[0][0] for pair in all_views}) == 1
        consensus = view_preds[0] if view_preds and len(set(view_preds)) == 1 else unknown
        valid = same_aug and same_teacher and consensus != unknown and min(view_confs) >= args.confidence
        original_id = labels.index(row["body_type"]) if row.get("body_type") in labels else unknown
        if valid and consensus == original_id:
            accepted += 1; accepted_classes[labels[consensus]] += 1; row["teacher_consensus"] = "accepted"; row["teacher_consensus_class"] = labels[consensus]
        else:
            if not same_aug: aug_conflict += 1
            if not same_teacher: teacher_conflict += 1
            if valid and consensus != original_id: original_mismatch += 1
            disagreement += 1; rejected_classes[row.get("body_type", "unknown")] += 1
            row["body_type"] = "unknown"; row["body_type_supervised"] = "false"; row["label_confidence"] = "unknown"; row["teacher_consensus"] = "rejected"; row["teacher_consensus_class"] = labels[consensus] if consensus != unknown else "unknown"
        row["teacher_consensus_confidence"] = f"{min(view_confs):.6f}"; row["teacher_consensus_views"] = ";".join(f"{name}:{labels[pair[0][0]]}/{labels[pair[1][0]]}" for (name, _, _), pair in zip(models, all_views))
    for row in rows:
        for key in ("teacher_consensus", "teacher_consensus_class", "teacher_consensus_confidence", "teacher_consensus_views"): row.setdefault(key, "not_audited")
    out_fields = fields + [x for x in ("teacher_consensus", "teacher_consensus_class", "teacher_consensus_confidence", "teacher_consensus_views") if x not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields); writer.writeheader(); writer.writerows(rows)
    report = {"schema_version": "body-teacher-consensus-v1", "input_manifest": str(args.manifest), "output_manifest": str(args.output_manifest), "input_manifest_sha256": sha(args.manifest), "audited_rows": len(target_indices), "accepted_rows": accepted, "rejected_rows": disagreement, "acceptance_rate": accepted / len(target_indices) if target_indices else 0.0, "augmentation_conflicts": aug_conflict, "teacher_conflicts": teacher_conflict, "accepted_class_counts": dict(accepted_classes), "rejected_original_class_counts": dict(rejected_classes), "original_label_mismatches_rejected": original_mismatch, "confidence_threshold": args.confidence, "models": {name: str(path) for name, path in checkpoints}, "frozen_video_used": False, "policy": "accept only when two views and all three teachers agree with the existing label; otherwise body_type=unknown and body_type_supervised=false"}
    args.output_report.parent.mkdir(parents=True, exist_ok=True); args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
