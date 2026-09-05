#!/usr/bin/env python3
"""Fail-closed color-label review using multiple teachers and three views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=/absolute/path.pt")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return name, path


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--minimum-total", type=int, default=100)
    parser.add_argument("--minimum-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two checkpoints are required")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite color teacher evidence")

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    colors = json.loads(args.labels.read_text(encoding="utf-8"))["colors"]
    color_index = {label: index for index, label in enumerate(colors)}
    selected = [
        index for index, row in enumerate(rows)
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and row.get("color") in color_index
        and row.get("color") != "unknown"
    ]
    dataset_root = args.dataset_root.resolve()

    class ReviewDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, local_index: int):
            row = rows[selected[local_index]]
            path = (dataset_root / row["image_path"]).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == "hflip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.80)).enhance(1.08)
            return self.transform(image), local_index

    device = torch.device(args.device)
    predictions: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint.get("colors", [])) != colors:
            raise RuntimeError(f"{name} color label order mismatch")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        views = ("original", "hflip", "dim")
        predictions[name] = [[None for _ in views] for _ in selected]
        for view_index, view in enumerate(views):
            loader = DataLoader(
                ReviewDataset(transform, view), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for local, pred, conf in zip(local_indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                        predictions[name][local][view_index] = (int(pred), float(conf))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"), "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accepted_counts = Counter()
    rejected_counts = Counter()
    reasons = Counter()
    confusion = Counter()
    accepted = 0
    review_fields = [
        "color_teacher_consensus", "color_teacher_class", "color_teacher_confidence",
        "color_teacher_views", "color_teacher_policy",
    ]
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        original = row["color"]
        pairs = [predictions[name][local_index] for name, _ in args.checkpoint]
        flat = [value for pair in pairs for value in pair if value is not None]
        predicted_ids = [value[0] for value in flat]
        confidences = [value[1] for value in flat]
        view_consistent = all(
            all(value is not None for value in pair)
            and len({value[0] for value in pair if value is not None}) == 1
            for pair in pairs
        )
        teacher_consistent = bool(predicted_ids) and len(set(predicted_ids)) == 1
        consensus = colors[predicted_ids[0]] if teacher_consistent else "unknown"
        minimum_confidence = min(confidences, default=0.0)
        passed = view_consistent and teacher_consistent and consensus == original and minimum_confidence >= args.confidence
        if passed:
            accepted += 1
            accepted_counts[original] += 1
            row["color_teacher_consensus"] = "accepted"
            row["review_status"] = "approved"
            row["color_review_status"] = "auto_foreground_multiteacher_three_view_consensus"
            row["formal_train_eligible"] = "true"
            row["label_confidence"] = "high"
            row["review_method"] = f"{row.get('review_method', '')}+multi_teacher_color_three_view_consensus"
            row["review_score"] = f"{minimum_confidence:.6f}"
        else:
            rejected_counts[original] += 1
            confusion[f"{original}->{consensus}"] += 1
            if not view_consistent:
                reasons["view_conflict"] += 1
            if not teacher_consistent:
                reasons["teacher_conflict"] += 1
            if teacher_consistent and consensus != original:
                reasons["source_label_mismatch"] += 1
            if minimum_confidence < args.confidence:
                reasons["confidence_below_threshold"] += 1
            row["color_teacher_consensus"] = "rejected"
            row["color"] = "unknown"
            row["color_supervised"] = "false"
            row["formal_train_eligible"] = "false"
            row["label_confidence"] = "unknown"
        row["color_teacher_class"] = consensus
        row["color_teacher_confidence"] = f"{minimum_confidence:.6f}"
        row["color_teacher_views"] = ";".join(
            f"{name}:" + "/".join(colors[value[0]] if value else "missing" for value in pair)
            for (name, _), pair in zip(args.checkpoint, pairs)
        )
        row["color_teacher_policy"] = "all teachers and original/hflip/dim views agree with foreground-proposed color at confidence floor"
    for row in rows:
        for field in review_fields:
            row.setdefault(field, "not_audited")
    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    status = "pass" if accepted >= args.minimum_total and len(accepted_counts) >= args.minimum_classes else "fail"
    report = {
        "schema_version": "color-multiteacher-consensus-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_manifest": str(args.manifest.resolve()), "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()), "output_manifest_sha256": sha256(args.output_manifest),
        "audited_rows": len(selected), "accepted_rows": accepted, "rejected_rows": len(selected) - accepted,
        "acceptance_rate": accepted / len(selected) if selected else 0.0,
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "rejected_original_color_counts": dict(sorted(rejected_counts.items())),
        "rejection_reasons_nonexclusive": dict(sorted(reasons.items())),
        "source_to_consensus_counts": dict(sorted(confusion.items())),
        "confidence_threshold": args.confidence,
        "views": ["original", "horizontal_flip", "brightness_0.80_contrast_1.08"],
        "minimum_total": args.minimum_total,
        "minimum_classes": args.minimum_classes,
        "checkpoints": checkpoint_evidence,
        "policy": {
            "foreground_or_official_labels_retained_only_not_replaced": True,
            "rejected_rows_changed_to_unknown_unsupervised": True,
            "model_predictions_used_to_create_new_labels": False,
            "test_split_not_used": True,
            "frozen_video_not_used": True,
            "production_model_unchanged": True,
            "deployment_not_performed": True,
        },
        "decision": "only pass-status accepted rows may enter a train-only color manifest",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
