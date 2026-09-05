#!/usr/bin/env python3
"""Build conservative train-only BMD-45 color labels without fake tracks.

All neural teachers and deterministic views must agree first.  A direct
foreground-pixel proposal must then independently agree with that consensus.
Rejected rows remain unknown and are not emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=/absolute/path.pt")
    name, raw = value.split("=", 1)
    path = Path(raw)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return name, path


def load_foreground_function():
    path = Path(__file__).with_name("build_bmd45_track_color_pseudolabels.py")
    spec = importlib.util.spec_from_file_location("stage52_foreground", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load foreground helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.foreground_color_evidence, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--teacher-confidence", type=float, default=0.80)
    parser.add_argument("--maximum-per-class", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two checkpoint entries are required")
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        all_rows = list(reader)
    target_indexes = [
        index for index, row in enumerate(all_rows)
        if row.get("split") == "train"
        and row.get("source_dataset") == "BMD-45-RAW"
        and row.get("source_license") == "CC-BY-4.0"
        and truthy(row.get("license_train_eligible"))
        and not truthy(row.get("color_supervised"))
        and row.get("color", "unknown") in {"", "unknown"}
    ]
    if not target_indexes:
        raise RuntimeError("no eligible BMD-45-RAW unknown-color train rows")
    dataset_root = args.manifest.resolve().parent

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    colors = json.loads(args.labels.read_text(encoding="utf-8"))["colors"]
    unknown_index = colors.index("unknown")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predictions: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence = {}

    class CandidateDataset(Dataset):
        def __init__(self, transform, view: int) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(target_indexes)

        def __getitem__(self, local_index: int):
            row = all_rows[target_indexes[local_index]]
            path = (dataset_root / row["image_path"]).resolve()
            path.relative_to(dataset_root.parent)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == 1:
                image = TF.hflip(image)
            elif self.view == 2:
                image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.80)).enhance(1.08)
            return self.transform(image), local_index

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
        predictions[name] = [[None, None, None] for _ in target_indexes]
        for view in range(3):
            loader = DataLoader(
                CandidateDataset(transform, view), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for local, pred, conf in zip(
                        local_indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()
                    ):
                        predictions[name][local][view] = (int(pred), float(conf))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    neural_candidates: list[tuple[int, str, float]] = []
    neural_reasons = Counter()
    neural_counts = Counter()
    for local_index, row_index in enumerate(target_indexes):
        flat: list[tuple[int, float]] = []
        view_consistent = True
        for name, _ in args.checkpoint:
            values = predictions[name][local_index]
            if any(value is None for value in values):
                view_consistent = False
                continue
            present = [value for value in values if value is not None]
            flat.extend(present)
            view_consistent &= len({value[0] for value in present}) == 1
        teacher_consistent = bool(flat) and len({value[0] for value in flat}) == 1
        minimum_confidence = min((value[1] for value in flat), default=0.0)
        predicted_index = flat[0][0] if teacher_consistent else unknown_index
        predicted_label = colors[predicted_index]
        if not view_consistent:
            neural_reasons["view_conflict"] += 1
        if not teacher_consistent:
            neural_reasons["teacher_conflict"] += 1
        if predicted_index == unknown_index:
            neural_reasons["predicted_unknown"] += 1
        if minimum_confidence < args.teacher_confidence:
            neural_reasons["confidence_below_floor"] += 1
        if (
            view_consistent and teacher_consistent and predicted_index != unknown_index
            and minimum_confidence >= args.teacher_confidence
        ):
            neural_candidates.append((row_index, predicted_label, minimum_confidence))
            neural_counts[predicted_label] += 1

    foreground_color_evidence, foreground_helper = load_foreground_function()
    accepted_by_class: dict[str, list[tuple[int, float, float, dict[str, object]]]] = {
        color: [] for color in colors if color != "unknown"
    }
    foreground_reasons = Counter()
    foreground_confusion = Counter()
    for row_index, teacher_label, minimum_confidence in neural_candidates:
        row = all_rows[row_index]
        path = (dataset_root / row["image_path"]).resolve()
        evidence = foreground_color_evidence(path)
        proposal = str(evidence.get("label", "unknown"))
        foreground_confusion[f"{teacher_label}->{proposal}"] += 1
        if proposal == "unknown":
            foreground_reasons[str(evidence.get("reason", "unknown"))] += 1
            continue
        if proposal != teacher_label:
            foreground_reasons["foreground_teacher_disagreement"] += 1
            continue
        accepted_by_class.setdefault(teacher_label, []).append(
            (row_index, minimum_confidence, float(evidence["score"]), evidence)
        )

    accepted: list[tuple[str, int, float, float, dict[str, object]]] = []
    for label, items in sorted(accepted_by_class.items()):
        items.sort(key=lambda item: (-item[1], -item[2], item[0]))
        accepted.extend((label, *item) for item in items[: args.maximum_per_class])

    output_rows = []
    for label, row_index, minimum_confidence, foreground_score, evidence in accepted:
        row = dict(all_rows[row_index])
        row.update({
            "color": label,
            "color_supervised": "true",
            "track_group": f"BMD45-independent:{Path(row['image_path']).stem}",
            "annotation_source": "bmd45_independent_foreground_multiteacher_v1",
            "color_review_status": "auto_foreground_multiteacher_consensus",
            "review_method": "all_teachers_original_flip_dim_then_foreground_pixel_agreement",
            "review_score": f"{min(minimum_confidence, foreground_score):.6f}",
            "review_model": ";".join(name for name, _ in args.checkpoint),
            "formal_train_eligible": "true",
            "pseudo_label": "true",
            "pseudo_label_confidence": f"{minimum_confidence:.6f}",
            "stage52_foreground_score": f"{foreground_score:.6f}",
            "stage52_foreground_margin": f"{float(evidence['margin']):.6f}",
            "stage52_mean_value": f"{float(evidence.get('mean_value', 0.0)):.6f}",
            "stage52_mean_saturation": f"{float(evidence.get('mean_saturation', 0.0)):.6f}",
        })
        output_rows.append(row)

    extras = [
        "stage52_foreground_score", "stage52_foreground_margin",
        "stage52_mean_value", "stage52_mean_saturation",
    ]
    fields = base_fields + [field for field in extras if field not in base_fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    accepted_counts = Counter(row["color"] for row in output_rows)
    low_light_rows = sum(float(row["stage52_mean_value"]) < 75 for row in output_rows)
    small_rows = sum(row.get("vehicle_size") == "small" for row in output_rows)
    report = {
        "schema_version": "bmd45-independent-color-pseudolabel-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if output_rows and len(accepted_counts) >= 4 else "fail",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "foreground_helper": str(foreground_helper.resolve()),
        "foreground_helper_sha256": sha256(foreground_helper),
        "source_dataset": "BMD-45-RAW",
        "source_license": "CC-BY-4.0",
        "eligible_unknown_color_train_rows": len(target_indexes),
        "neural_consensus_candidates": len(neural_candidates),
        "neural_consensus_color_counts": dict(sorted(neural_counts.items())),
        "neural_rejection_reasons_nonexclusive": dict(sorted(neural_reasons.items())),
        "foreground_rejection_reasons": dict(sorted(foreground_reasons.items())),
        "foreground_teacher_confusion": dict(sorted(foreground_confusion.items())),
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "low_light_proxy_rows": low_light_rows,
        "small_target_rows": small_rows,
        "checkpoints": checkpoint_evidence,
        "parameters": {
            "teacher_confidence": args.teacher_confidence,
            "views": ["original", "horizontal_flip", "brightness_0.80_contrast_1.08"],
            "maximum_per_class": args.maximum_per_class,
        },
        "policy": {
            "independent_images_not_claimed_as_tracks": True,
            "all_teachers_and_views_must_agree": True,
            "foreground_pixels_must_independently_agree": True,
            "teachers_cannot_replace_foreground_proposal": True,
            "rejected_rows_remain_unknown_unsupervised": True,
            "train_split_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only pass-status rows may enter a separated Stage52 color manifest after agent visual contact-sheet review",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible": len(target_indexes),
        "neural_candidates": len(neural_candidates),
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "output_manifest_sha256": report["output_manifest_sha256"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
