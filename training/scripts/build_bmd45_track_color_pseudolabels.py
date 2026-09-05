#!/usr/bin/env python3
"""Create conservative BMD-45 train-only color pseudo labels from true tracks.

The proposed color is derived from foreground pixels, then audited by temporal
agreement, multiple neural checkpoints, and three deterministic views.  Neural
teachers may reject a proposal but never invent or replace its color label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint  # noqa: E402


COLORS = [
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige",
]


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


def evenly_spaced(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return rows
    indexes = sorted({round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)})
    return [rows[index] for index in indexes]


def foreground_color_evidence(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"label": "unknown", "reason": "unreadable", "score": 0.0, "margin": 0.0}
    height, width = image.shape[:2]
    if min(height, width) < 18:
        return {"label": "unknown", "reason": "tiny_crop", "score": 0.0, "margin": 0.0}
    # GrabCut cost grows quickly with source resolution.  Color evidence does
    # not need fine edges, so use a deterministic bounded working resolution.
    maximum_dimension = max(height, width)
    if maximum_dimension > 128:
        scale = 128.0 / maximum_dimension
        image = cv2.resize(
            image, (max(18, round(width * scale)), max(18, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    inset_x = max(1, round(width * 0.065))
    inset_y = max(1, round(height * 0.065))
    rect = (inset_x, inset_y, max(1, width - 2 * inset_x), max(1, height - 2 * inset_y))
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, mask, rect, background, foreground, 1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return {"label": "unknown", "reason": "grabcut_failed", "score": 0.0, "margin": 0.0}
    vehicle = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    geometry = np.zeros_like(vehicle)
    geometry[round(height * 0.08): max(round(height * 0.84), 1),
             round(width * 0.08): max(round(width * 0.92), 1)] = True
    vehicle &= geometry
    minimum_pixels = max(180, round(height * width * 0.08))
    if int(vehicle.sum()) < minimum_pixels:
        return {
            "label": "unknown", "reason": "insufficient_foreground", "score": 0.0,
            "margin": 0.0, "foreground_fraction": float(vehicle.mean()),
        }

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][vehicle]
    saturation = hsv[:, :, 1][vehicle]
    value = hsv[:, :, 2][vehicle]
    total = float(len(hue))
    chromatic = (saturation >= 48) & (value >= 35)

    def fraction(condition: np.ndarray) -> float:
        return float(np.count_nonzero(condition) / total)

    scores = {
        "red": fraction(chromatic & ((hue <= 10) | (hue >= 170))),
        "yellow_orange": fraction(chromatic & (hue > 10) & (hue < 38)),
        "green": fraction(chromatic & (hue >= 38) & (hue < 90)),
        "blue": fraction(chromatic & (hue >= 90) & (hue < 140)),
        "brown_beige": fraction(
            (hue >= 4) & (hue < 30) & (saturation >= 25) & (saturation < 190)
            & (value >= 35) & (value < 215)
        ),
        "black": fraction((value < 62) | ((value < 82) & (saturation < 125))),
        "white": fraction((saturation < 42) & (value > 182)),
        "silver_gray": fraction((saturation < 58) & (value >= 72) & (value <= 182)),
    }
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    label, score = ordered[0]
    runner_up = ordered[1][1]
    margin = score - runner_up
    chromatic_labels = {"red", "yellow_orange", "green", "blue", "brown_beige"}
    minimum_score = 0.16 if label in chromatic_labels else 0.34
    minimum_margin = 0.045 if label in chromatic_labels else 0.075
    accepted = score >= minimum_score and margin >= minimum_margin
    return {
        "label": label if accepted else "unknown",
        "proposed_label": label,
        "reason": "accepted" if accepted else "weak_or_ambiguous_color_pixels",
        "score": score,
        "margin": margin,
        "runner_up_score": runner_up,
        "foreground_fraction": float(vehicle.mean()),
        "mean_value": float(value.mean()),
        "mean_saturation": float(saturation.mean()),
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--teacher-confidence", type=float, default=0.60)
    parser.add_argument("--track-evidence-agreement", type=float, default=0.80)
    parser.add_argument("--minimum-evidence-frames", type=int, default=3)
    parser.add_argument("--audit-frames-per-track", type=int, default=5)
    parser.add_argument("--output-frames-per-track", type=int, default=3)
    parser.add_argument("--maximum-per-class", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two independent checkpoint entries are required")
    if args.audit_frames_per_track < args.minimum_evidence_frames:
        raise ValueError("audit frames per track must cover the evidence minimum")
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    with args.tracks.open("r", encoding="utf-8-sig", newline="") as handle:
        track_rows = [row for row in csv.DictReader(handle) if truthy(row.get("track_valid"))]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in track_rows:
        groups[row["track_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["frame_number"]))

    evidence_by_key: dict[tuple[str, str], dict[str, object]] = {}
    proposal_tracks: dict[str, tuple[str, list[dict[str, str]]]] = {}
    rejection_reasons = Counter()
    proposed_counts = Counter()
    for track_id, rows in sorted(groups.items()):
        chosen = evenly_spaced(rows, args.audit_frames_per_track)
        evidence = []
        for row in chosen:
            item = foreground_color_evidence(Path(row["image_path"]))
            evidence_by_key[(track_id, row["annotation_id"])] = item
            if item["label"] != "unknown":
                evidence.append((row, item))
        if len(evidence) < args.minimum_evidence_frames:
            rejection_reasons["insufficient_foreground_evidence_frames"] += 1
            continue
        counts = Counter(str(item["label"]) for _, item in evidence)
        label, count = counts.most_common(1)[0]
        if count / len(evidence) < args.track_evidence_agreement:
            rejection_reasons["foreground_track_label_conflict"] += 1
            continue
        matching = [row for row, item in evidence if item["label"] == label]
        if len(matching) < args.minimum_evidence_frames:
            rejection_reasons["insufficient_majority_evidence_frames"] += 1
            continue
        proposal_tracks[track_id] = (label, matching)
        proposed_counts[label] += 1

    audit_rows: list[tuple[str, str, dict[str, str]]] = []
    for track_id, (label, rows) in sorted(proposal_tracks.items()):
        for row in rows:
            audit_rows.append((track_id, label, row))

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    expected_colors = json.loads(args.labels.read_text(encoding="utf-8"))["colors"]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    teacher_results: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence = {}

    class AuditDataset(Dataset):
        def __init__(self, transform, view: int) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(audit_rows)

        def __getitem__(self, index: int):
            path = Path(audit_rows[index][2]["image_path"])
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == 1:
                image = TF.hflip(image)
            elif self.view == 2:
                image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.80)).enhance(1.08)
            return self.transform(image), index

    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint.get("colors", [])) != expected_colors:
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
        teacher_results[name] = [[None, None, None] for _ in audit_rows]
        for view in range(3):
            loader = DataLoader(
                AuditDataset(transform, view), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for index, pred, conf in zip(indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                        teacher_results[name][index][view] = (int(pred), float(conf))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    passed_by_track: dict[str, list[tuple[dict[str, str], float, float]]] = defaultdict(list)
    teacher_reasons = Counter()
    for index, (track_id, proposed_label, row) in enumerate(audit_rows):
        flattened: list[tuple[int, float]] = []
        view_consistent = True
        for name, _ in args.checkpoint:
            values = teacher_results[name][index]
            if any(value is None for value in values):
                view_consistent = False
                continue
            present = [value for value in values if value is not None]
            flattened.extend(present)
            view_consistent &= len({value[0] for value in present}) == 1
        teacher_consistent = bool(flattened) and len({value[0] for value in flattened}) == 1
        predicted_label = expected_colors[flattened[0][0]] if teacher_consistent else "unknown"
        minimum_confidence = min((value[1] for value in flattened), default=0.0)
        if not view_consistent:
            teacher_reasons["view_conflict"] += 1
        if not teacher_consistent:
            teacher_reasons["teacher_conflict"] += 1
        if teacher_consistent and predicted_label != proposed_label:
            teacher_reasons["teacher_foreground_disagreement"] += 1
        if minimum_confidence < args.teacher_confidence:
            teacher_reasons["teacher_confidence_below_floor"] += 1
        if (
            view_consistent and teacher_consistent and predicted_label == proposed_label
            and minimum_confidence >= args.teacher_confidence
        ):
            foreground = evidence_by_key[(track_id, row["annotation_id"])]
            passed_by_track[track_id].append((row, minimum_confidence, float(foreground["score"])))

    accepted_by_class: dict[str, list[tuple[str, dict[str, str], float, float]]] = defaultdict(list)
    rejected_tracks = Counter()
    for track_id, (label, _) in sorted(proposal_tracks.items()):
        passed = passed_by_track.get(track_id, [])
        if len(passed) < args.minimum_evidence_frames:
            rejected_tracks["insufficient_teacher_approved_frames"] += 1
            continue
        passed.sort(key=lambda item: (item[1] + item[2], float(item[0]["bbox_area_ratio"])), reverse=True)
        for row, teacher_confidence, foreground_score in passed[: args.output_frames_per_track]:
            accepted_by_class[label].append((track_id, row, teacher_confidence, foreground_score))

    accepted: list[tuple[str, str, dict[str, str], float, float]] = []
    for label, items in sorted(accepted_by_class.items()):
        items.sort(key=lambda item: (-item[2], -item[3], item[0], int(item[1]["frame_number"])))
        accepted.extend((label, *item) for item in items[: args.maximum_per_class])

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        base_rows = list(reader)
    base_by_name = {
        Path(row["image_path"]).name: row for row in base_rows
        if row.get("split") == "train" and row.get("source_dataset") == "BMD-45-RAW"
    }
    output_rows = []
    missing_base_rows = []
    for label, track_id, track_row, teacher_confidence, foreground_score in accepted:
        name = Path(track_row["image_path"]).name
        base = base_by_name.get(name)
        if base is None:
            if len(missing_base_rows) < 20:
                missing_base_rows.append(name)
            continue
        row = dict(base)
        row.update({
            "color": label,
            "color_supervised": "true",
            "track_group": track_id,
            "annotation_source": "bmd45_foreground_temporal_teacher_consensus_v1",
            "source_license": "CC-BY-4.0",
            "license_train_eligible": "true",
            "review_status": "approved",
            "color_review_status": "auto_foreground_track_consensus",
            "review_method": "foreground_pixels_then_temporal_multi_teacher_three_view_audit",
            "review_score": f"{min(teacher_confidence, foreground_score):.6f}",
            "review_model": ";".join(name for name, _ in args.checkpoint),
            "formal_train_eligible": "true",
            "pseudo_label": "true",
            "pseudo_label_confidence": f"{teacher_confidence:.6f}",
            "stage52_foreground_score": f"{foreground_score:.6f}",
            "stage52_track_length": track_row["track_length"],
            "stage52_frame_number": track_row["frame_number"],
        })
        output_rows.append(row)

    extra_fields = ["stage52_foreground_score", "stage52_track_length", "stage52_frame_number"]
    fields = base_fields + [field for field in extra_fields if field not in base_fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    accepted_counts = Counter(row["color"] for row in output_rows)
    accepted_tracks = len({row["track_group"] for row in output_rows})
    low_light_rows = sum(
        float(evidence_by_key.get((row["track_group"], Path(row["image_path"]).stem.split("_")[-1]), {}).get("mean_value", 256)) < 75
        for row in output_rows
    )
    report = {
        "schema_version": "bmd45-track-color-pseudolabel-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if output_rows and len(accepted_counts) >= 4 else "fail",
        "tracks": str(args.tracks.resolve()),
        "tracks_sha256": sha256(args.tracks),
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "source_dataset": "BMD-45",
        "source_license": "CC-BY-4.0",
        "valid_input_tracks": len(groups),
        "foreground_proposal_tracks": len(proposal_tracks),
        "foreground_proposed_color_counts": dict(sorted(proposed_counts.items())),
        "foreground_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "teacher_audited_frames": len(audit_rows),
        "teacher_rejection_reasons_nonexclusive": dict(sorted(teacher_reasons.items())),
        "teacher_rejected_tracks": dict(sorted(rejected_tracks.items())),
        "accepted_tracks": accepted_tracks,
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "low_light_proxy_rows": low_light_rows,
        "missing_base_row_examples": missing_base_rows,
        "checkpoints": checkpoint_evidence,
        "parameters": {
            "teacher_confidence": args.teacher_confidence,
            "track_evidence_agreement": args.track_evidence_agreement,
            "minimum_evidence_frames": args.minimum_evidence_frames,
            "audit_frames_per_track": args.audit_frames_per_track,
            "output_frames_per_track": args.output_frames_per_track,
            "maximum_per_class": args.maximum_per_class,
        },
        "policy": {
            "foreground_pixels_create_the_proposal": True,
            "teachers_only_accept_or_reject": True,
            "multi_frame_consistency_required": True,
            "original_flip_and_dim_view_consistency_required": True,
            "train_split_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only pass-status rows may be appended to a separated Stage52 color training manifest after agent visual contact-sheet review",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "valid_input_tracks": len(groups),
        "foreground_proposal_tracks": len(proposal_tracks),
        "accepted_tracks": accepted_tracks,
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
