#!/usr/bin/env python3
"""Aggregate validation-only error attribution for the body truck router."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

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
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    index10 = min(len(ordered) - 1, int(0.10 * (len(ordered) - 1)))
    index90 = min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "p10": ordered[index10],
        "p90": ordered[index90],
    }


def compact_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def checkpoint_transform(checkpoint: dict[str, Any], transforms: Any):
    """Build the exact checkpoint-declared image geometry for diagnostics."""
    size = int(checkpoint["input_size"])
    mode = checkpoint.get("resize_mode", "stretch")
    if mode == "stretch":
        operations = [transforms.Resize((size, size), antialias=True)]
    elif mode == "center_crop":
        operations = [
            transforms.Resize(round(size * 232 / 224), antialias=True),
            transforms.CenterCrop(size),
        ]
    elif mode == "letterbox":
        operations = [base.SquarePad(), transforms.Resize((size, size), antialias=True)]
    else:
        raise RuntimeError(f"unsupported resize mode: {mode}")
    return transforms.Compose(
        [
            *operations,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_truth[record["truth"]].append(record)
    truth_summary: dict[str, Any] = {}
    for truth, items in sorted(by_truth.items()):
        family_items = [item for item in items if item["choose_family"]]
        truth_summary[truth] = {
            "support": len(items),
            "choose_family": len(family_items),
            "choose_family_rate": len(family_items) / len(items) if items else 0.0,
            "main_argmax": compact_counter(Counter(item["main_argmax"] for item in items)),
            "routed_at_050": compact_counter(Counter(item["routed"] for item in items)),
            "specialist_argmax_when_family": compact_counter(
                Counter(item["specialist_argmax"] for item in family_items)
            ),
            "family_mass": distribution([item["family_mass"] for item in items]),
            "specialist_best_probability_when_family": distribution(
                [item["specialist_best_probability"] for item in family_items]
            ),
        }
    family_records = [record for record in records if record["choose_family"]]
    subtype_records = [
        record
        for record in family_records
        if record["truth"] in {"light_truck", "heavy_truck"}
    ]
    subtype_correct = sum(
        record["specialist_argmax"] == record["truth"] for record in subtype_records
    )
    light_predictions = [
        record for record in family_records if record["specialist_argmax"] == "light_truck"
    ]
    light_true_positive = sum(record["truth"] == "light_truck" for record in light_predictions)
    light_truth_family = [
        record for record in family_records if record["truth"] == "light_truck"
    ]
    light_recalled = sum(
        record["specialist_argmax"] == "light_truck" for record in light_truth_family
    )
    light_family_rate = (
        len(light_truth_family) / len(by_truth.get("light_truck", []))
        if by_truth.get("light_truck")
        else 0.0
    )
    light_specialist_recall = (
        light_recalled / len(light_truth_family) if light_truth_family else 0.0
    )
    light_specialist_precision = (
        light_true_positive / len(light_predictions) if light_predictions else 0.0
    )
    if light_family_rate < 0.80 and light_specialist_recall < 0.80:
        diagnosis = "main_family_router_and_specialist_domain_repair_required"
    elif light_family_rate < 0.80:
        diagnosis = "main_family_router_domain_repair_required"
    elif light_specialist_recall < 0.80 or light_specialist_precision < 0.80:
        diagnosis = "light_heavy_specialist_domain_repair_required"
    else:
        diagnosis = "calibration_or_abstention_policy_is_primary_gap"
    return {
        "records": len(records),
        "by_truth": truth_summary,
        "truck_family_route": {
            "chosen_rows": len(family_records),
            "subtype_truth_rows": len(subtype_records),
            "specialist_argmax_accuracy_on_subtype_truth": (
                subtype_correct / len(subtype_records) if subtype_records else 0.0
            ),
            "light_truth_family_route_rate": light_family_rate,
            "light_specialist_recall_given_family_route": light_specialist_recall,
            "light_specialist_precision_given_family_route": light_specialist_precision,
        },
        "diagnosis": diagnosis,
    }


def assert_hash(path: Path, expected: str, role: str) -> None:
    actual = base.sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")


def main() -> int:
    args = parse_args()
    for path, expected, role in (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.labels, args.expected_labels_sha256, "labels"),
        (args.body_checkpoint, args.expected_body_checkpoint_sha256, "body checkpoint"),
        (
            args.specialist_checkpoint,
            args.expected_specialist_checkpoint_sha256,
            "specialist checkpoint",
        ),
    ):
        assert_hash(path, expected, role)
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = base.resolve_rows(args.manifest, args.datasets_safety_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    body_model, body_checkpoint = base.load_model(args.body_checkpoint, device)
    specialist_model, specialist_checkpoint = base.load_model(args.specialist_checkpoint, device)
    base.validate_checkpoint_taxonomy(body_checkpoint, body_checkpoint, labels)
    base.validate_truck_specialist_checkpoint(specialist_checkpoint)
    body_labels = body_checkpoint["body_types"]
    specialist_labels = specialist_checkpoint["body_types"]
    family_indices = torch.tensor(
        [body_labels.index(label) for label in ("truck", "light_truck", "heavy_truck")],
        device=device,
    )
    outside_mask = torch.ones(len(body_labels), device=device, dtype=torch.bool)
    outside_mask[family_indices] = False
    body_transform = checkpoint_transform(body_checkpoint, transforms)
    specialist_transform = checkpoint_transform(specialist_checkpoint, transforms)

    class Images(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            with Image.open(rows[index]["_resolved_image_path"]) as image:
                rgb = image.convert("RGB")
                return body_transform(rgb), specialist_transform(rgb), index

    loader = DataLoader(
        Images(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, specialist_images, indexes in loader:
            images = images.to(device, non_blocking=True)
            specialist_images = specialist_images.to(device, non_blocking=True)
            body_logits, _ = body_model(images)
            specialist_logits, _ = specialist_model(specialist_images)
            body_probabilities = torch.softmax(body_logits, dim=1)
            main_indices = body_probabilities.argmax(dim=1)
            family_mass = body_probabilities.index_select(1, family_indices).sum(dim=1)
            outside_best = body_probabilities[:, outside_mask].amax(dim=1)
            choose_family = family_mass > outside_best
            specialist_probabilities = torch.softmax(specialist_logits, dim=1)
            specialist_best_probability, specialist_indices = specialist_probabilities[:, :2].max(dim=1)
            routed_indices, _ = base.route_body_with_specialist(
                body_logits,
                body_labels,
                specialist_logits,
                specialist_labels,
                0.50,
            )
            for position, index in enumerate(indexes.tolist()):
                if not base.truthy(rows[index].get("body_type_supervised")):
                    continue
                records.append(
                    {
                        "truth": rows[index]["body_type"],
                        "lighting": rows[index].get("lighting") or "unknown",
                        "vehicle_size": rows[index].get("vehicle_size") or "unknown",
                        "main_argmax": body_labels[int(main_indices[position])],
                        "family_mass": float(family_mass[position]),
                        "outside_best": float(outside_best[position]),
                        "choose_family": bool(choose_family[position]),
                        "specialist_argmax": specialist_labels[int(specialist_indices[position])],
                        "specialist_best_probability": float(
                            specialist_best_probability[position]
                        ),
                        "routed": body_labels[int(routed_indices[position])],
                    }
                )
    aggregate = summarize(records)
    light_scene: dict[str, Any] = {}
    for field in ("lighting", "vehicle_size"):
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["truth"] == "light_truck":
                groups[record[field]].append(record)
        light_scene[field] = {
            group: summarize(items)["truck_family_route"]
            for group, items in sorted(groups.items())
        }
    report = {
        "schema_version": "stage111-body-router-error-analysis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only_error_attribution",
        "inputs": {
            "manifest_sha256": base.sha256(args.manifest),
            "labels_sha256": base.sha256(args.labels),
            "body_checkpoint_sha256": base.sha256(args.body_checkpoint),
            "specialist_checkpoint_sha256": base.sha256(args.specialist_checkpoint),
            "geometry_contract": {
                "body": {
                    "input_size": int(body_checkpoint["input_size"]),
                    "resize_mode": body_checkpoint.get("resize_mode", "stretch"),
                },
                "specialist": {
                    "input_size": int(specialist_checkpoint["input_size"]),
                    "resize_mode": specialist_checkpoint.get("resize_mode", "stretch"),
                },
            },
        },
        "aggregate": aggregate,
        "light_truck_stratified": light_scene,
        "policy": {
            "split": "validation",
            "aggregate_only_no_sample_paths_emitted": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        f"{base.sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], **aggregate["truck_family_route"], "diagnosis": aggregate["diagnosis"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
