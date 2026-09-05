#!/usr/bin/env python3
"""Retain VTID2 labels only when their unseen-fold teacher is view-consistent."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-prefix", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--minimum-total", type=int, default=500)
    parser.add_argument("--minimum-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite OOF audit evidence")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    manifest_paths = [Path(f"{args.manifest_prefix}.fold{fold}.csv").resolve() for fold in range(args.folds)]
    if any(not path.is_file() for path in manifest_paths):
        raise FileNotFoundError("one or more OOF manifests are missing")
    with manifest_paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    row_keys = [(row["image_path"], row["body_type"], row["oof_fold"]) for row in rows]
    fold_role_verified = True
    for fold, path in enumerate(manifest_paths):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        if [(row["image_path"], row["body_type"], row["oof_fold"]) for row in fold_rows] != row_keys:
            raise RuntimeError(f"row identity/order differs in fold {fold}")
        if any((int(row["oof_fold"]) == fold) != (row["split"] == "audit") for row in fold_rows):
            fold_role_verified = False
    if not fold_role_verified:
        raise RuntimeError("audit fold role verification failed")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))["body_types"]
    label_index = {label: index for index, label in enumerate(labels)}
    dataset_root = args.dataset_root.resolve()
    predictions: list[list[tuple[int, float] | None]] = [[None, None] for _ in rows]
    checkpoint_evidence = []
    fold_metrics = []

    class AuditDataset(Dataset):
        def __init__(self, indexes: list[int], transform, flip: bool) -> None:
            self.indexes = indexes
            self.transform = transform
            self.flip = flip

        def __len__(self) -> int:
            return len(self.indexes)

        def __getitem__(self, local_index: int):
            row_index = self.indexes[local_index]
            path = (dataset_root / rows[row_index]["image_path"]).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.flip:
                image = TF.hflip(image)
            return self.transform(image), row_index

    device = torch.device(args.device)
    for fold in range(args.folds):
        checkpoint_path = (args.run_root / f"fold{fold}" / "best.pt").resolve()
        metrics_path = (args.run_root / f"fold{fold}" / "metrics.json").resolve()
        test_metrics_path = (args.run_root / f"fold{fold}" / "test_metrics.json").resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint.get("body_types", [])) != labels:
            raise RuntimeError(f"fold {fold} label order mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        if test_metrics.get("status") != "not_run":
            raise RuntimeError(f"fold {fold} did not preserve the test-skip policy")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 256))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        indexes = [index for index, row in enumerate(rows) if int(row["oof_fold"]) == fold]
        for view_index, flip in enumerate((False, True)):
            loader = DataLoader(
                AuditDataset(indexes, transform, flip), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, row_indexes in loader:
                    body_logits, _ = model(images.to(device, non_blocking=True))
                    confidence, predicted = body_logits.softmax(dim=1).max(dim=1)
                    for row_index, pred, conf in zip(row_indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                        predictions[row_index][view_index] = (int(pred), float(conf))
        checkpoint_evidence.append({
            "fold": fold, "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
            "input_size": input_size, "architecture": checkpoint.get("architecture"),
            "nested_validation_metrics": str(metrics_path), "nested_validation_metrics_sha256": sha256(metrics_path),
            "test_metrics": str(test_metrics_path), "test_metrics_sha256": sha256(test_metrics_path),
        })
        fold_metrics.append({"fold": fold, "audit_rows": len(indexes)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accepted_counts = Counter()
    rejected_counts = Counter()
    reasons = Counter()
    confusion = Counter()
    fold_acceptance = Counter()
    accepted = 0
    review_fields = [
        "oof_teacher_consensus", "oof_teacher_class", "oof_teacher_confidence",
        "oof_teacher_views", "oof_teacher_policy",
    ]
    for row_index, row in enumerate(rows):
        original = row["body_type"]
        pair = predictions[row_index]
        if pair[0] is None or pair[1] is None:
            raise RuntimeError(f"missing OOF prediction for row {row_index}")
        view_consistent = pair[0][0] == pair[1][0]
        predicted = labels[pair[0][0]] if view_consistent else "unknown"
        minimum_confidence = min(pair[0][1], pair[1][1])
        passed = view_consistent and predicted == original and minimum_confidence >= args.confidence
        if passed:
            accepted += 1
            accepted_counts[original] += 1
            fold_acceptance[int(row["oof_fold"])] += 1
            row["oof_teacher_consensus"] = "accepted"
            row["review_method"] = f"{row.get('review_method', '')}+unseen_fold_teacher_original_hflip_consensus"
            row["review_score"] = f"{minimum_confidence:.6f}"
        else:
            rejected_counts[original] += 1
            confusion[f"{original}->{predicted}"] += 1
            if not view_consistent:
                reasons["view_conflict"] += 1
            if view_consistent and predicted != original:
                reasons["source_label_mismatch"] += 1
            if minimum_confidence < args.confidence:
                reasons["confidence_below_threshold"] += 1
            row["oof_teacher_consensus"] = "rejected"
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["formal_train_eligible"] = "false"
            row["label_confidence"] = "unknown"
        row["oof_teacher_class"] = predicted
        row["oof_teacher_confidence"] = f"{minimum_confidence:.6f}"
        row["oof_teacher_views"] = f"{labels[pair[0][0]]}/{labels[pair[1][0]]}"
        row["oof_teacher_policy"] = "teacher never trained or early-stopped on this fold; original and horizontal flip must agree with official label"
    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    all_classes_present = all(accepted_counts[label] >= args.minimum_per_class for label in sorted(label_index) if label in {"sedan", "suv", "pickup"})
    status = "pass" if accepted >= args.minimum_total and all_classes_present else "fail"
    report = {
        "schema_version": "vtid2-oof-teacher-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest_prefix": args.manifest_prefix,
        "manifest_sha256_by_fold": [sha256(path) for path in manifest_paths],
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "audited_rows": len(rows),
        "accepted_rows": accepted,
        "rejected_rows": len(rows) - accepted,
        "acceptance_rate": accepted / len(rows) if rows else 0.0,
        "accepted_class_counts": dict(sorted(accepted_counts.items())),
        "rejected_original_class_counts": dict(sorted(rejected_counts.items())),
        "rejection_reasons_nonexclusive": dict(sorted(reasons.items())),
        "source_to_oof_teacher_counts": dict(sorted(confusion.items())),
        "fold_accepted_counts": {str(key): value for key, value in sorted(fold_acceptance.items())},
        "confidence_threshold": args.confidence,
        "minimum_total": args.minimum_total,
        "minimum_per_class": args.minimum_per_class,
        "fold_role_verified": fold_role_verified,
        "checkpoints": checkpoint_evidence,
        "policy": {
            "audit_row_unseen_by_its_teacher_train": True,
            "audit_row_unseen_by_its_teacher_nested_validation": True,
            "official_labels_retained_only_not_replaced": True,
            "rejected_rows_changed_to_unknown_unsupervised": True,
            "test_split_not_used": True,
            "frozen_video_not_used": True,
            "production_model_unchanged": True,
            "deployment_not_performed": True,
        },
        "decision": "only pass-status accepted rows may become a train-only supplement; independent VFG/UA validation remains mandatory",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
