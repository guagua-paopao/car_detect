#!/usr/bin/env python3
"""Assign body subtypes to unlabeled real hard crops using strict diverse-teacher consensus."""

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


COARSE_COMPATIBILITY = {
    "car": {"sedan", "suv", "mpv", "van", "pickup"},
    "bus": {"bus"},
    "truck": {"light_truck", "heavy_truck", "pickup"},
}
VIEWS = ("original", "horizontal_flip", "dim")


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
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 3:
        raise ValueError("at least three diverse checkpoints are required")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite hard-crop teacher evidence")

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
    if "unknown" not in labels:
        raise RuntimeError("body label ontology has no unknown class")
    selected = [
        index for index, row in enumerate(rows)
        if row.get("split") == "train"
        and not truthy(row.get("body_type_supervised"))
        and row.get("body_type") == "unknown"
        and row.get("official_vehicle_class") in COARSE_COMPATIBILITY
    ]
    if not selected:
        raise RuntimeError("no unlabeled hard train crops selected")
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
            if self.view == "horizontal_flip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = TF.adjust_contrast(TF.adjust_brightness(image, 0.72), 1.08)
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
        predictions[name] = [[None for _ in VIEWS] for _ in selected]
        for view_index, view in enumerate(VIEWS):
            loader = DataLoader(
                ReviewDataset(transform, view),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indices in loader:
                    body_logits, _ = model(images.to(device, non_blocking=True))
                    confidence, predicted = body_logits.softmax(dim=1).max(dim=1)
                    for local, pred, conf in zip(
                        local_indices.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()
                    ):
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
    accepted_official_counts = Counter()
    accepted_condition_counts = Counter()
    rejected_reasons = Counter()
    consensus_counts = Counter()
    accepted = 0
    review_fields = [
        "teacher_consensus", "teacher_consensus_class", "teacher_consensus_confidence",
        "teacher_consensus_views", "teacher_audit_policy",
    ]
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        pairs = [predictions[name][local_index] for name, _ in args.checkpoint]
        flat = [value for teacher in pairs for value in teacher if value is not None]
        predicted_ids = [value[0] for value in flat]
        confidences = [value[1] for value in flat]
        per_teacher_view_consistent = all(
            all(value is not None for value in teacher)
            and len({value[0] for value in teacher if value is not None}) == 1
            for teacher in pairs
        )
        all_teachers_consistent = bool(predicted_ids) and len(set(predicted_ids)) == 1
        consensus_label = labels[predicted_ids[0]] if all_teachers_consistent else "unknown"
        minimum_confidence = min(confidences, default=0.0)
        coarse_compatible = consensus_label in COARSE_COMPATIBILITY[row["official_vehicle_class"]]
        passed = (
            per_teacher_view_consistent
            and all_teachers_consistent
            and coarse_compatible
            and consensus_label != "unknown"
            and minimum_confidence >= args.confidence
        )
        consensus_counts[consensus_label] += 1
        if passed:
            accepted += 1
            accepted_counts[consensus_label] += 1
            accepted_official_counts[row["official_vehicle_class"]] += 1
            for condition in ("small_target_proxy", "occluded", "truncated"):
                source_field = "vehicle_size" if condition == "small_target_proxy" else condition
                is_present = row.get(source_field) == "small" if condition == "small_target_proxy" else truthy(row.get(source_field))
                if is_present:
                    accepted_condition_counts[condition] += 1
            row["body_type"] = consensus_label
            row["body_type_supervised"] = "true"
            row["formal_train_eligible"] = "true"
            row["review_status"] = "accepted_multiteacher_pseudolabel"
            row["label_confidence"] = "high"
            row["review_method"] = "diverse_four_teacher_three_view_exact_consensus"
            row["review_score"] = f"{minimum_confidence:.6f}"
            row["pseudo_label_confidence"] = f"{minimum_confidence:.6f}"
            row["teacher_consensus"] = "accepted"
        else:
            if not per_teacher_view_consistent:
                rejected_reasons["augmentation_conflict"] += 1
            if not all_teachers_consistent:
                rejected_reasons["teacher_conflict"] += 1
            if all_teachers_consistent and not coarse_compatible:
                rejected_reasons["official_coarse_class_conflict"] += 1
            if consensus_label == "unknown":
                rejected_reasons["unknown_consensus"] += 1
            if minimum_confidence < args.confidence:
                rejected_reasons["confidence_below_threshold"] += 1
            row["teacher_consensus"] = "rejected"
            row["review_status"] = "rejected_to_unknown"
            row["formal_train_eligible"] = "false"
        row["teacher_consensus_class"] = consensus_label
        row["teacher_consensus_confidence"] = f"{minimum_confidence:.6f}"
        row["teacher_consensus_views"] = ";".join(
            f"{name}:" + "/".join(labels[value[0]] if value else "missing" for value in teacher)
            for (name, _), teacher in zip(args.checkpoint, pairs)
        )
        row["teacher_audit_policy"] = (
            "all diverse teachers agree across original/hflip/dim views, confidence floor, "
            "and official coarse car/bus/truck compatibility"
        )
    for row in rows:
        for field in review_fields:
            row.setdefault(field, "not_audited")
    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    status = "pass" if accepted >= 1000 and len(accepted_counts) >= 5 else "fail"
    report = {
        "schema_version": "body-unlabeled-hard-multiteacher-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "audited_rows": len(selected),
        "accepted_rows": accepted,
        "rejected_rows": len(selected) - accepted,
        "acceptance_rate": accepted / len(selected),
        "accepted_body_type_counts": dict(sorted(accepted_counts.items())),
        "accepted_official_vehicle_counts": dict(sorted(accepted_official_counts.items())),
        "accepted_hard_condition_counts": dict(sorted(accepted_condition_counts.items())),
        "all_consensus_class_counts": dict(sorted(consensus_counts.items())),
        "rejection_reasons_nonexclusive": dict(sorted(rejected_reasons.items())),
        "confidence_threshold": args.confidence,
        "views": list(VIEWS),
        "coarse_compatibility": {key: sorted(value) for key, value in COARSE_COMPATIBILITY.items()},
        "checkpoints": checkpoint_evidence,
        "release_requirements": {"minimum_accepted": 1000, "minimum_body_classes": 5},
        "policy": {
            "all_teachers_exact_agreement": True,
            "all_views_exact_agreement": True,
            "official_coarse_class_compatibility_required": True,
            "rejected_rows_remain_unknown_unsupervised": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": (
            "accepted rows remain candidates pending perceptual deduplication, class/source caps "
            "and agent visual contact-sheet review; no training is authorized by this report alone"
        )
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
