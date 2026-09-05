#!/usr/bin/env python3
"""Revalidate the retained BMD-45 CCTV color seed with the Stage73 rule.

The input seed was selected by older pseudo-label teachers.  Its old color
value is never used for acceptance or replacement.  The SHA-pinned Stage73
validation rule, two current teachers and foreground pixels independently
decide each train-only image.  Exact and perceptual near duplicates are then
removed before any row can become auxiliary supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from scripts.build_bmd45_track_color_pseudolabels import foreground_color_evidence  # noqa: E402
from scripts.build_stage73_uadetrac_color_pseudolabels_v2 import (  # noqa: E402
    foreground_support,
    load_color_contract,
    load_validation_rule,
    parse_checkpoint,
    sha256,
    truthy,
)
from scripts.evaluate_stage73_color_pseudolabel_rule import (  # noqa: E402
    aggregate_teachers,
    foreground_compatible,
)
from src.multitask_mobilenet_v3 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    model_from_checkpoint,
)


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def deduplicate_accepts(
    accepts: list[dict[str, Any]], maximum_hamming: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ordered = sorted(
        accepts,
        key=lambda item: (
            -float(item["teacher_confidence"]),
            -float(item["foreground_support"]),
            str(item["row"].get("image_path", "")),
        ),
    )
    retained: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    seen_dhash: list[tuple[str, str]] = []
    reasons = Counter()
    for item in ordered:
        row = item["row"]
        digest = str(row.get("sha256", "")).lower()
        dhash = str(row.get("dhash64", "")).lower()
        if digest and digest in seen_sha:
            reasons["exact_sha256"] += 1
            continue
        near = next(
            (
                label
                for prior_hash, label in seen_dhash
                if dhash and len(dhash) == 16 and hamming(dhash, prior_hash) <= maximum_hamming
            ),
            None,
        )
        if near is not None:
            reasons[
                "near_duplicate_same_color"
                if near == item["predicted"]
                else "near_duplicate_cross_color_conflict"
            ] += 1
            continue
        retained.append(item)
        if digest:
            seen_sha.add(digest)
        if dhash and len(dhash) == 16:
            seen_dhash.append((dhash, str(item["predicted"])))
    return retained, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--validation-rule-report", type=Path, required=True)
    parser.add_argument("--expected-validation-rule-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-total", type=int, default=300)
    parser.add_argument("--minimum-classes", type=int, default=5)
    parser.add_argument("--minimum-rows-per-counted-class", type=int, default=20)
    parser.add_argument("--maximum-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two independently trained checkpoints are required")
    if sha256(args.manifest).lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError("BMD-45 seed manifest SHA256 mismatch")
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    manifest = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    if "test" in {part.lower() for part in manifest.parts}:
        raise RuntimeError("seed manifest path is not train-only")
    colors, labels_sha = load_color_contract(args.labels, args.expected_labels_sha256)
    rule, rule_sha = load_validation_rule(
        args.validation_rule_report, args.expected_validation_rule_sha256
    )
    known_colors = set(colors) - {"unknown", "other"}

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        source_rows = list(reader)
    eligible: list[dict[str, str]] = []
    filter_reasons = Counter()
    old_color_counts = Counter()
    for row in source_rows:
        if row.get("split") != "train":
            filter_reasons["non_train"] += 1
            continue
        if row.get("source_dataset") != "BMD-45-RAW":
            filter_reasons["wrong_source"] += 1
            continue
        if row.get("source_license") != "CC-BY-4.0" or not truthy(row.get("license_train_eligible")):
            filter_reasons["license_ineligible"] += 1
            continue
        raw = Path(row["image_path"])
        path = (dataset_root / raw).resolve()
        try:
            path.relative_to(dataset_root.parent)
        except ValueError:
            filter_reasons["outside_dataset_tree"] += 1
            continue
        if not path.is_file():
            filter_reasons["missing"] += 1
            continue
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (OSError, ValueError):
            filter_reasons["unreadable"] += 1
            continue
        candidate = dict(row)
        candidate["_resolved_path"] = str(path)
        eligible.append(candidate)
        old_color_counts[str(row.get("color", "unknown"))] += 1
        filter_reasons["eligible"] += 1
    if not eligible:
        raise RuntimeError("no readable train-only BMD-45 seed images")

    foreground: list[dict[str, Any]] = []
    for row in eligible:
        foreground.append(foreground_color_evidence(Path(row["_resolved_path"])))

    class SeedDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(eligible)

        def __getitem__(self, index: int):
            with Image.open(eligible[index]["_resolved_path"]) as opened:
                image = opened.convert("RGB")
            if self.view == "hflip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = ImageEnhance.Contrast(
                    ImageEnhance.Brightness(image).enhance(0.80)
                ).enhance(1.08)
            return self.transform(image), index

    views = ("original", "hflip", "dim")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predictions: dict[str, list[list[tuple[str, float]]]] = {}
    checkpoint_evidence: dict[str, dict[str, Any]] = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if list(checkpoint.get("colors", [])) != colors:
            raise RuntimeError(f"{name} color label order mismatch")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        predictions[name] = [[] for _ in eligible]
        for view in views:
            loader = DataLoader(
                SeedDataset(transform, view),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for index, prediction, score in zip(
                        indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()
                    ):
                        predictions[name][index].append((colors[int(prediction)], float(score)))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accepts: list[dict[str, Any]] = []
    rejection_reasons = Counter()
    old_new_confusion = Counter()
    for index, row in enumerate(eligible):
        teacher_views = {name: predictions[name][index] for name, _ in args.checkpoint}
        decision = aggregate_teachers(
            teacher_views,
            int(rule["minimum_views"]),
            float(rule["confidence_floor"]),
        )
        if decision is None:
            rejection_reasons["teacher_view_confidence_or_label_conflict"] += 1
            continue
        predicted, confidence = decision
        if predicted not in known_colors:
            rejection_reasons["predicted_non_target"] += 1
            continue
        evidence = foreground[index]
        if not foreground_compatible(
            evidence,
            predicted,
            str(rule["foreground_mode"]),
            float(rule["chromatic_floor"]),
            float(rule["achromatic_floor"]),
        ):
            rejection_reasons["foreground_incompatible"] += 1
            continue
        support = foreground_support(evidence, predicted)
        accepts.append({
            "row": row,
            "predicted": predicted,
            "teacher_confidence": float(confidence),
            "foreground_support": support,
        })
        old_new_confusion[f"{row.get('color', 'unknown')}->{predicted}"] += 1

    deduplicated, duplicate_reasons = deduplicate_accepts(
        accepts, args.near_duplicate_hamming
    )
    by_class: dict[str, list[dict[str, Any]]] = {
        color: [] for color in sorted(known_colors)
    }
    for item in deduplicated:
        by_class[item["predicted"]].append(item)
    retained = [
        item
        for color in sorted(by_class)
        for item in by_class[color][: args.maximum_per_class]
    ]

    output_rows: list[dict[str, str]] = []
    for item in retained:
        row = {key: value for key, value in item["row"].items() if key != "_resolved_path"}
        row.update({
            "color": str(item["predicted"]),
            "color_supervised": "true",
            "annotation_source": "bmd45_stage74_validation_frozen_revalidation",
            "review_status": "approved",
            "color_review_status": "auto_stage73_rule_revalidated",
            "review_method": "validation-frozen two-teacher view consensus+foreground compatibility",
            "review_score": f"{min(item['teacher_confidence'], item['foreground_support']):.6f}",
            "review_model": ";".join(name for name, _ in args.checkpoint),
            "pseudo_label": "true",
            "pseudo_label_confidence": f"{item['teacher_confidence']:.6f}",
            "stage74_foreground_score": f"{item['foreground_support']:.6f}",
            "stage74_validation_rule_sha256": rule_sha,
        })
        output_rows.append(row)
    extra_fields = ["stage74_foreground_score", "stage74_validation_rule_sha256"]
    output_fields = fields + [field for field in extra_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(row["color"] for row in output_rows)
    counted_classes = sum(
        count >= args.minimum_rows_per_counted_class for count in counts.values()
    )
    status = (
        "pass_revalidated_auxiliary_available"
        if len(output_rows) >= args.minimum_total and counted_classes >= args.minimum_classes
        else "fail_closed_insufficient_revalidated_auxiliary"
    )
    report = {
        "schema_version": "stage74-bmd45-seed-color-revalidation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "eligibility": "auxiliary_research_only_pending_joint_audit",
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256(manifest),
        "source_dataset": "BMD-45-RAW",
        "source_license": "CC-BY-4.0",
        "validation_rule_report": str(args.validation_rule_report.resolve()),
        "validation_rule_report_sha256": rule_sha,
        "validation_frozen_parameters": rule,
        "labels_sha256": labels_sha,
        "input_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "old_color_counts_not_used_for_acceptance": dict(sorted(old_color_counts.items())),
        "accepted_before_dedup": len(accepts),
        "duplicate_rejections": dict(sorted(duplicate_reasons.items())),
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(counts.items())),
        "classes_with_minimum_rows": counted_classes,
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "old_new_confusion_audit_only": dict(sorted(old_new_confusion.items())),
        "checkpoints": checkpoint_evidence,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "gates": {
            "minimum_total": args.minimum_total,
            "minimum_classes": args.minimum_classes,
            "minimum_rows_per_counted_class": args.minimum_rows_per_counted_class,
            "near_duplicate_hamming": args.near_duplicate_hamming,
        },
        "policy": {
            "old_pseudo_color_used_for_selection_or_acceptance": False,
            "old_pseudo_color_used_for_posthoc_audit_only": True,
            "validation_report_used_for_fixed_parameters_only": True,
            "foreground_pixels_only_accept_or_reject": True,
            "train_images_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only pass may enter a later joint UA+BMD train-only leakage and quota audit",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "accepted_rows": len(output_rows), "counts": counts}))
    return 0 if status.startswith("pass_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
