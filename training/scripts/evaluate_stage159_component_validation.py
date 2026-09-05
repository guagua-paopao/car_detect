#!/usr/bin/env python3
"""Validate one fresh body or color candidate on Stage159 validation only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import evaluate_stage109_color_class_thresholds as thresholding
import evaluate_v2_decoupled_shared_validation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("body", "color"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--expected-production-config-sha256", required=True)
    parser.add_argument("--body-specialist-checkpoint", type=Path)
    parser.add_argument("--expected-body-specialist-sha256")
    parser.add_argument("--body-specialist-subtype-threshold", type=float, default=0.50)
    parser.add_argument("--resolution-root", type=Path, required=True)
    parser.add_argument("--allowed-image-root", type=Path, action="append", required=True)
    parser.add_argument("--precision-target", type=float, default=0.935)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = base.resolve_rows(args.manifest, args.resolution_root)
    roots = [path.resolve() for path in args.allowed_image_root]
    for row in rows:
        path = Path(row["_resolved_image_path"]).resolve()
        if not any(path == root or root in path.parents for root in roots):
            raise RuntimeError(f"validation image outside explicit roots: {path}")
    return rows


def supervised(row: dict[str, str], head: str) -> bool:
    return base.truthy(row.get("body_type_supervised" if head == "body" else "color_supervised"))


def label_key(head: str) -> str:
    return "body_type" if head == "body" else "color"


def prediction_key(head: str) -> str:
    return f"{head}_label"


def confidence_key(head: str) -> str:
    return f"{head}_confidence"


def fine_color_top_label(labels: list[str], probabilities: list[float]) -> tuple[str, float]:
    """Preserve v2 fine labels; legacy v1 checkpoints remain merged by design."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("color labels/probabilities mismatch")
    index = max(range(len(labels)), key=probabilities.__getitem__)
    return labels[index], float(probabilities[index])


def production_truth_label(head: str, truth: str) -> str:
    """Map v2 validation truth into the deployed v1 taxonomy for baseline scoring.

    Coverage comparisons still use the deployed model's actual emitted/unknown
    decision.  This mapping prevents a correct legacy merged color from being
    counted as an error merely because the candidate taxonomy is finer.
    """
    if head != "color":
        return truth
    return {
        "gray": "silver_gray",
        "silver": "silver_gray",
        "yellow": "yellow_orange",
        "brown": "brown_beige",
    }.get(truth, truth)


def validate_color_checkpoint_taxonomy(
    checkpoint: dict[str, Any], expected_labels: dict[str, Any] | None
) -> None:
    """Fail closed while scoping checks to the head being evaluated.

    Color-only candidates intentionally disable the body hierarchy, so asking
    them to satisfy the body truck-family contract would reject a valid color
    checkpoint before inference.  The body evaluator still uses the shared
    full taxonomy/hierarchy validation through ``base.infer_pair``.
    """
    if expected_labels is not None:
        if (
            checkpoint.get("labels_version") != expected_labels.get("labels_version")
            or checkpoint.get("colors") != expected_labels.get("colors")
        ):
            raise RuntimeError("candidate checkpoint taxonomy mismatch")
        return
    # The deployed baseline is a coupled v1 checkpoint; retain its complete
    # shared validation, including both body and color taxonomy.
    base.validate_checkpoint_taxonomy(checkpoint, checkpoint, None)


def infer_color_fine(args, rows, checkpoint_path, expected_labels):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = base.load_model(checkpoint_path, device)
    validate_color_checkpoint_taxonomy(checkpoint, expected_labels)
    size = int(checkpoint["input_size"])
    mode = checkpoint.get("resize_mode", "stretch")
    if mode == "stretch":
        operations = [transforms.Resize((size, size), antialias=True)]
    elif mode == "center_crop":
        operations = [transforms.Resize(round(size * 232 / 224), antialias=True), transforms.CenterCrop(size)]
    elif mode == "letterbox":
        operations = [base.SquarePad(), transforms.Resize((size, size), antialias=True)]
    else:
        raise RuntimeError(f"unsupported resize mode: {mode}")
    transform = transforms.Compose([
        *operations,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class Images(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(rows[index]["_resolved_image_path"]) as image:
                return transform(image.convert("RGB")), index

    loader = DataLoader(Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    outputs = [None] * len(rows)
    with torch.no_grad():
        for images, indexes in loader:
            images = images.to(device, non_blocking=True)
            _, logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            for position, raw_index in enumerate(indexes.tolist()):
                label, confidence = fine_color_top_label(checkpoint["colors"], probabilities[position])
                outputs[raw_index] = {
                    "body_label": "unknown",
                    "body_confidence": 0.0,
                    "color_label": label,
                    "color_confidence": confidence,
                }
    if any(value is None for value in outputs):
        raise RuntimeError("fine-color inference produced incomplete outputs")
    return outputs


def static_metric(rows, outputs, indexes, head, thresholds):
    return thresholding.metric(
        [outputs[index][prediction_key(head)] for index in indexes],
        [outputs[index][confidence_key(head)] for index in indexes],
        [rows[index][label_key(head)] for index in indexes],
        thresholds,
    )


def track_metrics(rows, outputs, head, thresholds):
    tracks: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not supervised(row, head):
            continue
        group = (
            row.get("track_key")
            or row.get("track_id")
            or row.get("track_group")
            or row.get("stage159_validation_group")
            or row.get("stage157_group")
            or f"row:{index}"
        )
        tracks[group].append(index)
    final_labels, final_truths = [], []
    labels_by_track: dict[str, list[str]] = {}
    for track, indexes in tracks.items():
        observations = []
        for index in indexes:
            predicted = outputs[index][prediction_key(head)]
            confidence = outputs[index][confidence_key(head)]
            emitted = predicted if predicted != "unknown" and confidence >= thresholds.get(predicted, 1.01) else "unknown"
            observations.append((emitted, confidence, base.quality_weight(rows[index])))
        sequence = base.fuse_observations(observations)
        labels_by_track[track] = sequence
        final_labels.append(sequence[-1])
        final_truths.append(Counter(rows[index][label_key(head)] for index in indexes).most_common(1)[0][0])
    selected = [index for index, label in enumerate(final_labels) if label != "unknown"]
    correct = sum(final_labels[index] == final_truths[index] for index in selected)
    return {
        "track_final": {
            "evaluated": len(final_truths),
            "selected": len(selected),
            "correct": correct,
            "precision": correct / len(selected) if selected else 0.0,
            "coverage": len(selected) / len(final_truths) if final_truths else 0.0,
            "unknown_rate": 1.0 - len(selected) / len(final_truths) if final_truths else 1.0,
        },
        "stability": base.stability(labels_by_track),
        "window": 5,
        "minimum_share": 0.60,
        "minimum_margin": 0.15,
    }


def stratified(rows, outputs, head, thresholds):
    result = {}
    for field in ("lighting", "vehicle_size", "weather", "occlusion_level", "source_dataset", "source_video"):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if supervised(row, head):
                groups[row.get(field) or "unknown"].append(index)
        result[field] = {group: static_metric(rows, outputs, indexes, head, thresholds) for group, indexes in sorted(groups.items())}
    return result


def main() -> int:
    args = parse_args()
    for path, expected, role in (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.candidate_checkpoint, args.expected_candidate_sha256, "candidate checkpoint"),
        (args.baseline_checkpoint, args.expected_baseline_sha256, "baseline checkpoint"),
        (args.production_config, args.expected_production_config_sha256, "production configuration"),
    ):
        thresholding.assert_hash(path, expected, role)
    if bool(args.body_specialist_checkpoint) != bool(args.expected_body_specialist_sha256):
        raise RuntimeError("specialist path/SHA contract failed")
    if args.head == "color" and args.body_specialist_checkpoint:
        raise RuntimeError("color validation must not route through a body specialist")
    if args.body_specialist_checkpoint:
        thresholding.assert_hash(args.body_specialist_checkpoint, args.expected_body_specialist_sha256, "body specialist")
    rows = resolve_rows(args)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.head == "color":
        outputs = infer_color_fine(args, rows, args.candidate_checkpoint, labels)
        baseline_outputs = infer_color_fine(args, rows, args.baseline_checkpoint, None)
    else:
        outputs = base.infer_pair(
            args, rows, args.candidate_checkpoint, args.candidate_checkpoint,
            expected_labels=labels, body_specialist_checkpoint_path=args.body_specialist_checkpoint,
        )
        baseline_outputs = base.infer_pair(
            args, rows, args.baseline_checkpoint, args.baseline_checkpoint,
            expected_labels=None,
        )
    indexes = [index for index, row in enumerate(rows) if supervised(row, args.head)]
    complex_indexes = [index for index in indexes if base.is_complex_row(rows[index])]
    if not indexes or not complex_indexes:
        raise RuntimeError("validation view lacks supervised or complex rows")
    predictions = [outputs[index][prediction_key(args.head)] for index in indexes]
    confidences = [outputs[index][confidence_key(args.head)] for index in indexes]
    truths = [rows[index][label_key(args.head)] for index in indexes]
    selection = thresholding.optimize_thresholds(predictions, confidences, truths, args.precision_target)
    thresholds = selection["thresholds"]
    overall = static_metric(rows, outputs, indexes, args.head, thresholds)
    complex_metric = static_metric(rows, outputs, complex_indexes, args.head, thresholds)
    track = track_metrics(rows, outputs, args.head, thresholds)

    production_config = json.loads(args.production_config.read_text(encoding="utf-8"))
    if production_config.get("models", {}).get("vehicle_attribute", {}).get("artifact") != "vehicle-attr-agent-e-224":
        raise RuntimeError("production configuration does not reference the locked baseline artifact")
    if production_config.get("labels_version") != "vehicle-labels-v1":
        raise RuntimeError("production configuration labels version changed")
    threshold_key = "type_threshold" if args.head == "body" else "color_threshold"
    baseline_threshold = float(production_config.get("vehicle_analytics", {}).get(threshold_key, 0.0))
    if not 0.0 < baseline_threshold <= 1.0:
        raise ValueError("baseline threshold must be in (0, 1]")
    baseline_predictions = [baseline_outputs[index][prediction_key(args.head)] for index in indexes]
    baseline_confidences = [baseline_outputs[index][confidence_key(args.head)] for index in indexes]
    baseline_truths = [production_truth_label(args.head, truth) for truth in truths]
    baseline_overall = base.metric_at_threshold(
        baseline_predictions, baseline_confidences, baseline_truths, baseline_threshold
    )
    baseline_complex = base.metric_at_threshold(
        [baseline_outputs[index][prediction_key(args.head)] for index in complex_indexes],
        [baseline_outputs[index][confidence_key(args.head)] for index in complex_indexes],
        [production_truth_label(args.head, rows[index][label_key(args.head)]) for index in complex_indexes],
        baseline_threshold,
    )
    if args.head == "body":
        complex_improvement = complex_metric["coverage"] - baseline_complex["coverage"]
        gates = {
            "body_static_precision": overall["precision"] >= 0.93,
            "body_static_coverage": overall["coverage"] >= 0.45,
            "body_complex_coverage_gain": complex_improvement >= 0.15,
            "body_track_precision": track["track_final"]["precision"] >= 0.93,
            "body_track_coverage": track["track_final"]["coverage"] >= 0.45,
            "body_track_stability": track["stability"]["transition_stability"] >= 0.95,
        }
        comparison = {"body_complex_coverage_gain": complex_improvement}
    else:
        unknown_reduction = (
            (baseline_complex["unknown_rate"] - complex_metric["unknown_rate"]) / baseline_complex["unknown_rate"]
            if baseline_complex["unknown_rate"] > 0 else 0.0
        )
        gates = {
            "color_static_precision": overall["precision"] >= 0.93,
            "color_static_coverage": overall["coverage"] >= 0.25,
            "color_complex_unknown_reduction": unknown_reduction >= 0.20,
            "color_track_precision": track["track_final"]["precision"] >= 0.93,
            "color_track_coverage": track["track_final"]["coverage"] >= 0.25,
            "color_track_stability": track["stability"]["transition_stability"] >= 0.95,
        }
        comparison = {"color_complex_unknown_relative_reduction": unknown_reduction}
    report = {
        "schema_version": "stage159-component-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "head": args.head,
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": base.sha256(args.manifest),
            "labels_sha256": base.sha256(args.labels),
            "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
            "candidate_checkpoint_sha256": base.sha256(args.candidate_checkpoint),
            "baseline_checkpoint_sha256": base.sha256(args.baseline_checkpoint),
            "body_specialist_checkpoint_sha256": base.sha256(args.body_specialist_checkpoint) if args.body_specialist_checkpoint else None,
        },
        "support": {"overall": len(indexes), "complex": len(complex_indexes)},
        "selection": selection,
        "production_baseline_policy": {
            "threshold": baseline_threshold,
            "source": str(args.production_config.resolve()),
            "config_sha256": base.sha256(args.production_config),
            "artifact": "vehicle-attr-agent-e-224",
            "v2_truth_mapped_to_v1_merged_taxonomy": args.head == "color",
        },
        "static": {"candidate_overall": overall, "candidate_complex": complex_metric, "production_overall": baseline_overall, "production_complex": baseline_complex},
        "comparison": comparison,
        "per_class": thresholding.per_class_metrics(predictions, confidences, truths, thresholds),
        "stratified": stratified(rows, outputs, args.head, thresholds),
        "track_fusion": track,
        "gates": {"gates": gates, "all_pass": all(gates.values())},
        "decision": f"{args.head}_component_validation_pass" if all(gates.values()) else f"{args.head}_component_rejected_fail_closed",
        "policy": {
            "split": "validation",
            "test_accessed": False,
            "stage148_or_stage155_reused": False,
            "frozen_video_used": False,
            "labels_rewritten": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    if args.output.exists() or args.output.parent.exists():
        raise FileExistsError("refusing to overwrite Stage159 component evidence")
    args.output.parent.mkdir(parents=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(f"{base.sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "overall": overall, "complex": complex_metric, "comparison": comparison, "track": track, "gates": gates}, ensure_ascii=False))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
