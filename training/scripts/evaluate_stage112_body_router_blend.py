#!/usr/bin/env python3
"""Validation-only convex calibration of main and specialist subtype evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import evaluate_stage109_color_class_thresholds as thresholding
import evaluate_stage110_body_class_thresholds as body_thresholding
import evaluate_v2_decoupled_shared_validation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-body-checkpoint-sha256", required=True)
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-specialist-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-validation-report", type=Path, required=True)
    parser.add_argument("--expected-baseline-validation-report-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.93)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def blended_output(record: dict[str, Any], alpha: float) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not record["choose_family"]:
        return {
            "body_label": record["main_label"],
            "body_confidence": record["main_confidence"],
        }
    light = (1.0 - alpha) * record["main_light_conditional"] + alpha * record["specialist_light"]
    heavy = (1.0 - alpha) * record["main_heavy_conditional"] + alpha * record["specialist_heavy"]
    label, subtype_confidence = (
        ("light_truck", light) if light >= heavy else ("heavy_truck", heavy)
    )
    return {
        "body_label": label,
        "body_confidence": math.sqrt(max(0.0, record["family_mass"] * subtype_confidence)),
    }


def infer_components(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    body_model, body_checkpoint = base.load_model(args.body_checkpoint, device)
    specialist_model, specialist_checkpoint = base.load_model(args.specialist_checkpoint, device)
    base.validate_checkpoint_taxonomy(body_checkpoint, body_checkpoint, labels)
    base.validate_truck_specialist_checkpoint(specialist_checkpoint)
    if body_checkpoint["input_size"] != specialist_checkpoint["input_size"]:
        raise RuntimeError("body and specialist input sizes differ")
    body_labels = body_checkpoint["body_types"]
    family_indices = torch.tensor(
        [body_labels.index(label) for label in ("truck", "light_truck", "heavy_truck")],
        device=device,
    )
    light_index = body_labels.index("light_truck")
    heavy_index = body_labels.index("heavy_truck")
    outside_mask = torch.ones(len(body_labels), device=device, dtype=torch.bool)
    outside_mask[family_indices] = False
    transform = transforms.Compose(
        [
            transforms.Resize((body_checkpoint["input_size"], body_checkpoint["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class Images(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            with Image.open(rows[index]["_resolved_image_path"]) as image:
                return transform(image.convert("RGB")), index

    loader = DataLoader(
        Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, Any] | None] = [None] * len(rows)
    with torch.no_grad():
        for images, indexes in loader:
            images = images.to(device, non_blocking=True)
            body_logits, _ = body_model(images)
            specialist_logits, _ = specialist_model(images)
            probabilities = torch.softmax(body_logits, dim=1)
            main_confidence, main_index = probabilities.max(dim=1)
            family_mass = probabilities.index_select(1, family_indices).sum(dim=1)
            outside_best = probabilities[:, outside_mask].amax(dim=1)
            choose_family = family_mass > outside_best
            fine_sum = (probabilities[:, light_index] + probabilities[:, heavy_index]).clamp_min(1e-12)
            main_light = probabilities[:, light_index] / fine_sum
            main_heavy = probabilities[:, heavy_index] / fine_sum
            specialist_probabilities = torch.softmax(specialist_logits, dim=1)
            for position, index in enumerate(indexes.tolist()):
                records[index] = {
                    "main_label": body_labels[int(main_index[position])],
                    "main_confidence": float(main_confidence[position]),
                    "choose_family": bool(choose_family[position]),
                    "family_mass": float(family_mass[position]),
                    "main_light_conditional": float(main_light[position]),
                    "main_heavy_conditional": float(main_heavy[position]),
                    "specialist_light": float(specialist_probabilities[position, 0]),
                    "specialist_heavy": float(specialist_probabilities[position, 1]),
                }
    if any(record is None for record in records):
        raise RuntimeError("component inference did not cover all rows")
    return [record for record in records if record is not None]


def main() -> int:
    args = parse_args()
    for path, expected, role in (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256, "body checkpoint"),
        (args.specialist_checkpoint, args.expected_specialist_checkpoint_sha256, "specialist checkpoint"),
        (
            args.baseline_validation_report,
            args.expected_baseline_validation_report_sha256,
            "baseline validation report",
        ),
    ):
        thresholding.assert_hash(path, expected, role)
    rows = base.resolve_rows(args.manifest, args.datasets_safety_root)
    indexes = [index for index, row in enumerate(rows) if base.truthy(row.get("body_type_supervised"))]
    if not indexes or any(not base.is_complex_row(rows[index]) for index in indexes):
        raise RuntimeError("expected a non-empty all-complex supervised body validation view")
    truths = [rows[index]["body_type"] for index in indexes]
    components = infer_components(args, rows)
    baseline_report = json.loads(args.baseline_validation_report.read_text(encoding="utf-8"))
    if baseline_report.get("policy", {}).get("split") != "validation":
        raise RuntimeError("pinned baseline report is not validation-only")
    if baseline_report.get("policy", {}).get("test_accessed") is not False:
        raise RuntimeError("pinned baseline report accessed test")
    baseline_static = baseline_report["static"]["production_baseline"]

    trials: list[dict[str, Any]] = []
    trial_payloads: dict[float, tuple[list[dict[str, Any]], dict[str, float]]] = {}
    for step in range(21):
        alpha = step / 20.0
        outputs = [blended_output(record, alpha) for record in components]
        predictions = [outputs[index]["body_label"] for index in indexes]
        confidences = [outputs[index]["body_confidence"] for index in indexes]
        selection = thresholding.optimize_thresholds(
            predictions, confidences, truths, args.precision_target
        )
        thresholds = selection["thresholds"]
        static = thresholding.metric(predictions, confidences, truths, thresholds)
        track = body_thresholding.fused(rows, outputs, thresholds)
        gain = static["coverage"] - baseline_static["coverage"]
        gates = {
            "body_static_precision": static["precision"] >= 0.93,
            "body_static_coverage": static["coverage"] >= 0.45,
            "body_complex_coverage_gain": gain >= 0.15,
            "body_track_precision": track["track_final"]["precision"] >= 0.93,
            "body_track_coverage": track["track_final"]["coverage"] >= 0.45,
            "body_track_stability": track["stability"]["transition_stability"] >= 0.95,
        }
        trials.append(
            {
                "alpha_specialist": alpha,
                "thresholds": thresholds,
                "static": static,
                "complex_coverage_gain": gain,
                "track_final": track["track_final"],
                "stability": track["stability"],
                "gates": gates,
                "all_pass": all(gates.values()),
            }
        )
        trial_payloads[alpha] = (outputs, thresholds)
    passing = [trial for trial in trials if trial["all_pass"]]
    selected = max(
        passing or trials,
        key=lambda trial: (
            int(trial["all_pass"]),
            sum(trial["gates"].values()),
            trial["static"]["coverage"],
            trial["track_final"]["coverage"],
            trial["static"]["precision"],
        ),
    )
    outputs, thresholds = trial_payloads[selected["alpha_specialist"]]
    predictions = [outputs[index]["body_label"] for index in indexes]
    confidences = [outputs[index]["body_confidence"] for index in indexes]
    report = {
        "schema_version": "stage112-body-router-blend-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "inputs": {
            "manifest_sha256": base.sha256(args.manifest),
            "labels_sha256": base.sha256(args.labels),
            "body_checkpoint_sha256": base.sha256(args.body_checkpoint),
            "specialist_checkpoint_sha256": base.sha256(args.specialist_checkpoint),
            "baseline_validation_report_sha256": base.sha256(args.baseline_validation_report),
        },
        "selection": {
            "source": "validation only",
            "precision_target": args.precision_target,
            "selected": selected,
            "qualified_alphas": [trial["alpha_specialist"] for trial in passing],
            "trials": trials,
        },
        "per_class": thresholding.per_class_metrics(
            predictions, confidences, truths, thresholds
        ),
        "stratified": body_thresholding.stratified(rows, outputs, thresholds),
        "decision": (
            "body_blend_validation_pass_pending_integrated_gate"
            if selected["all_pass"]
            else "body_blend_rejected_fail_closed"
        ),
        "policy": {
            "split": "validation",
            "test_accessed": False,
            "frozen_video_used": False,
            "labels_rewritten": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        f"{base.sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"decision": report["decision"], "qualified_alphas": report["selection"]["qualified_alphas"], "selected": selected}, ensure_ascii=False))
    return 0 if selected["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
