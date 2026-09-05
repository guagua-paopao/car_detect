#!/usr/bin/env python3
"""Conservatively relabel BMD generic Truck crops with specialist/view consensus."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint  # noqa: E402
from build_stage120_axle_semantic_clean_manifest import post_split_leaks, sha256, truthy  # noqa: E402


VIEWS = ("original", "horizontal_flip", "dim")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=/absolute/path.pt")
    name, raw = value.split("=", 1)
    path = Path(raw)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return name, path


def decide_consensus(
    teacher_views: list[list[tuple[str, float]]], confidence: float
) -> tuple[str | None, str, float]:
    flat = [value for views in teacher_views for value in views]
    if not flat or any(len(views) != len(VIEWS) for views in teacher_views):
        return None, "missing_view", 0.0
    if any(len({label for label, _ in views}) != 1 for views in teacher_views):
        return None, "augmentation_conflict", min(score for _, score in flat)
    labels = {label for label, _ in flat}
    minimum = min(score for _, score in flat)
    if len(labels) != 1:
        return None, "teacher_conflict", minimum
    label = next(iter(labels))
    if label not in {"light_truck", "heavy_truck"}:
        return None, "unknown_consensus", minimum
    if minimum < confidence:
        return None, "confidence_below_threshold", minimum
    return label, "accepted", minimum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 3:
        raise ValueError("at least three specialist checkpoints are required")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage125 evidence")
    for path, expected, role in (
        (args.source_manifest, args.expected_source_manifest_sha256, "source manifest"),
        (args.base_manifest, args.expected_base_manifest_sha256, "base manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
    ):
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with args.source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_reader = csv.DictReader(handle)
        source_fields = list(source_reader.fieldnames or [])
        source_rows = list(source_reader)
    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        base_reader = csv.DictReader(handle)
        base_fields = list(base_reader.fieldnames or [])
        base_rows = list(base_reader)
    if any(row.get("split") == "test" for row in source_rows + base_rows):
        raise RuntimeError("test rows are forbidden")
    if any(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in source_rows + base_rows
    ):
        raise RuntimeError("frozen marker found")
    selected = [
        row for row in source_rows
        if row.get("split") == "train"
        and row.get("source_dataset") in {"BMD-45", "BMD-45-RAW"}
        and row.get("body_type") == "heavy_truck"
        and not truthy(row.get("research_only"))
        and "NC" not in row.get("source_license", "").upper()
    ]
    if len(selected) < 10000:
        raise RuntimeError("too few BMD generic Truck candidates")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["body_types"]
    if labels != ["light_truck", "heavy_truck", "unknown"]:
        raise RuntimeError("specialist labels contract mismatch")
    dataset_root = args.dataset_root.resolve()

    class ReviewDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, index: int):
            raw = Path(selected[index]["image_path"])
            path = raw.resolve() if raw.is_absolute() else (args.source_manifest.parent / raw).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == "horizontal_flip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = TF.adjust_contrast(TF.adjust_brightness(image, 0.72), 1.08)
            return self.transform(image), index

    device = torch.device(args.device)
    predictions: dict[str, list[list[tuple[str, float] | None]]] = {}
    checkpoint_evidence = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("body_types") != labels:
            raise RuntimeError(f"{name} specialist labels mismatch")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 256))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        predictions[name] = [[None for _ in VIEWS] for _ in selected]
        for view_index, view in enumerate(VIEWS):
            loader = DataLoader(
                ReviewDataset(transform, view), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, indices in loader:
                    logits, _ = model(images.to(device, non_blocking=True))
                    probabilities = logits.softmax(dim=1)
                    confidence_values, predicted = probabilities.max(dim=1)
                    for index, pred, score in zip(indices.tolist(), predicted.cpu().tolist(), confidence_values.cpu().tolist()):
                        predictions[name][index][view_index] = (labels[pred], float(score))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"), "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accepted_rows = []
    accepted_counts = Counter()
    rejection_reasons = Counter()
    confidence_buckets = Counter()
    for index, row in enumerate(selected):
        teacher_views = []
        view_evidence = []
        for name, _ in args.checkpoint:
            values = [value for value in predictions[name][index] if value is not None]
            teacher_views.append(values)
            view_evidence.append(f"{name}:" + "/".join(label for label, _ in values))
        label, reason, minimum = decide_consensus(teacher_views, args.confidence)
        confidence_buckets[f"{int(minimum * 10) / 10:.1f}"] += 1
        if label is None:
            rejection_reasons[reason] += 1
            continue
        output = dict(row)
        output["body_type"] = label
        output["body_type_supervised"] = "true"
        output["pseudo_label"] = "true"
        output["pseudo_label_confidence"] = f"{minimum:.6f}"
        output["review_status"] = "accepted_multiteacher_pseudolabel"
        output["review_method"] = "three_specialists_three_views_exact_consensus"
        output["sample_weight"] = "0.350000"
        output["stage125_original_body_type"] = row.get("body_type", "")
        output["stage125_consensus_label"] = label
        output["stage125_consensus_min_confidence"] = f"{minimum:.6f}"
        output["stage125_consensus_views"] = ";".join(view_evidence)
        output["stage125_truth_role"] = "bmd_generic_truck_conservative_pseudolabel"
        accepted_rows.append(output)
        accepted_counts[label] += 1

    merged = [*base_rows, *accepted_rows]
    leaks = post_split_leaks(merged, 4)
    if any(leaks.values()):
        raise RuntimeError(f"Stage125 cross-split leak: {leaks}")
    output_fields = list(dict.fromkeys([
        *base_fields, *source_fields, "stage125_original_body_type", "stage125_consensus_label",
        "stage125_consensus_min_confidence", "stage125_consensus_views", "stage125_truth_role",
    ]))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=False)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(merged)
    status = "pass" if accepted_counts["light_truck"] >= 200 and accepted_counts["heavy_truck"] >= 500 else "fail"
    report = {
        "schema_version": "stage125-bmd-generic-truck-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "inputs": {"source_manifest_sha256": sha256(args.source_manifest), "base_manifest_sha256": sha256(args.base_manifest)},
        "audited_rows": len(selected), "accepted_rows": len(accepted_rows),
        "accepted_counts": dict(sorted(accepted_counts.items())),
        "rejected_rows": len(selected) - len(accepted_rows),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "minimum_confidence_distribution": dict(sorted(confidence_buckets.items())),
        "confidence_threshold": args.confidence, "views": list(VIEWS), "checkpoints": checkpoint_evidence,
        "output": {"manifest": str(args.output_manifest.resolve()), "manifest_sha256": sha256(args.output_manifest), "rows": len(merged)},
        "integrity": {"post_split_leaks": leaks, "frozen_rows": 0},
        "policy": {
            "all_teachers_exact_agreement": True, "all_views_exact_agreement": True,
            "uncertain_rows_excluded_not_coerced": True, "pseudo_label_weight": 0.35,
            "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_manifest, args.output_report):
        Path(str(path) + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": status, "audited": len(selected), "accepted": dict(accepted_counts), "rejected": dict(rejection_reasons)}, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
