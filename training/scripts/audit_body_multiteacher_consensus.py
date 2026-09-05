#!/usr/bin/env python3
"""Fail-closed body-label review using multiple teachers and deterministic views."""

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
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--minimum-total", type=int, default=300)
    parser.add_argument("--minimum-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two independent checkpoints are required")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite teacher-audit evidence")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["body_types"]
    label_index = {label: index for index, label in enumerate(labels)}
    selected = [
        index for index, row in enumerate(rows)
        if row.get("split") == "train"
        and truthy(row.get("body_type_supervised"))
        and row.get("body_type") in label_index
        and row.get("body_type") != "unknown"
    ]
    dataset_root = args.dataset_root.resolve()

    class ReviewDataset(Dataset):
        def __init__(self, transform, flip: bool) -> None:
            self.transform = transform
            self.flip = flip

        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, local_index: int):
            row = rows[selected[local_index]]
            path = (dataset_root / row["image_path"]).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.flip:
                image = TF.hflip(image)
            return self.transform(image), local_index

    device = torch.device(args.device)
    predictions: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint.get("body_types", [])) != labels:
            raise RuntimeError(f"{name} body label order does not match labels file")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        predictions[name] = [[None, None] for _ in selected]
        for view_index, flip in enumerate((False, True)):
            loader = DataLoader(
                ReviewDataset(transform, flip), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indices in loader:
                    logits, _ = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for local, pred, conf in zip(local_indices.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                        predictions[name][local][view_index] = (int(pred), float(conf))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "input_size": input_size,
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
        "teacher_consensus", "teacher_consensus_class", "teacher_consensus_confidence",
        "teacher_consensus_views", "teacher_audit_policy",
    ]
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        original = row["body_type"]
        teacher_pairs = [predictions[name][local_index] for name, _ in args.checkpoint]
        flat = [value for pair in teacher_pairs for value in pair if value is not None]
        predicted_ids = [value[0] for value in flat]
        confidences = [value[1] for value in flat]
        view_consistent = all(pair[0] is not None and pair[1] is not None and pair[0][0] == pair[1][0] for pair in teacher_pairs)
        teacher_consistent = bool(predicted_ids) and len(set(predicted_ids)) == 1
        consensus_id = predicted_ids[0] if teacher_consistent else label_index["unknown"]
        consensus_label = labels[consensus_id]
        minimum_confidence = min(confidences, default=0.0)
        matches_source = consensus_label == original
        passed = (
            view_consistent and teacher_consistent and matches_source
            and consensus_label != "unknown" and minimum_confidence >= args.confidence
        )
        if passed:
            accepted += 1
            accepted_counts[original] += 1
            row["teacher_consensus"] = "accepted"
            row["review_method"] = f"{row.get('review_method', '')}+multi_teacher_two_view_consensus"
            row["review_score"] = f"{minimum_confidence:.6f}"
        else:
            rejected_counts[original] += 1
            confusion[f"{original}->{consensus_label}"] += 1
            if not view_consistent:
                reasons["augmentation_conflict"] += 1
            if not teacher_consistent:
                reasons["teacher_conflict"] += 1
            if teacher_consistent and not matches_source:
                reasons["source_label_mismatch"] += 1
            if minimum_confidence < args.confidence:
                reasons["confidence_below_threshold"] += 1
            row["teacher_consensus"] = "rejected"
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["formal_train_eligible"] = "false"
            row["label_confidence"] = "unknown"
        row["teacher_consensus_class"] = consensus_label
        row["teacher_consensus_confidence"] = f"{minimum_confidence:.6f}"
        row["teacher_consensus_views"] = ";".join(
            f"{name}:{labels[pair[0][0]] if pair[0] else 'missing'}/{labels[pair[1][0]] if pair[1] else 'missing'}"
            for (name, _), pair in zip(args.checkpoint, teacher_pairs)
        )
        row["teacher_audit_policy"] = "all teachers and original/hflip views agree with official label at confidence floor"
    for row in rows:
        for field in review_fields:
            row.setdefault(field, "not_audited")
    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": "body-multiteacher-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if accepted >= args.minimum_total and len(accepted_counts) >= args.minimum_classes else "fail",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "audited_rows": len(selected),
        "accepted_rows": accepted,
        "rejected_rows": len(selected) - accepted,
        "acceptance_rate": accepted / len(selected) if selected else 0.0,
        "accepted_class_counts": dict(sorted(accepted_counts.items())),
        "rejected_original_class_counts": dict(sorted(rejected_counts.items())),
        "rejection_reasons_nonexclusive": dict(sorted(reasons.items())),
        "source_to_consensus_counts": dict(sorted(confusion.items())),
        "confidence_threshold": args.confidence,
        "minimum_total": args.minimum_total,
        "minimum_classes": args.minimum_classes,
        "views": ["original", "horizontal_flip"],
        "checkpoints": checkpoint_evidence,
        "policy": {
            "official_labels_retained_only_on_full_consensus": True,
            "rejected_rows_changed_to_unknown_unsupervised": True,
            "model_predictions_used_to_create_new_labels": False,
            "frozen_video_used": False,
            "test_split_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only accepted rows may be appended to a train-only candidate manifest; rejected rows remain unknown and unsupervised",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
