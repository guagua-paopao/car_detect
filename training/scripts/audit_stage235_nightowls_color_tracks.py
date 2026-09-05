#!/usr/bin/env python3
"""Fail-closed NightOwls color pseudo-label audit using two teachers and tracks.

Only official training pixels are read.  A color becomes supervised when all
teacher/view predictions agree on enough frames of one track.  At most three
non-near-duplicate frames per accepted track are admitted so contiguous video
frames cannot inflate the effective sample count.
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


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    model_from_checkpoint,
)


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


def number(value: str | None, default: float = 0.0) -> float:
    try:
        parsed = float(value or "")
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def frame_confidence_floor(row: dict[str, str], base: float) -> float:
    floor = base
    size_bin = row.get("size_bin", "")
    if size_bin == "small":
        floor += 0.10
    elif size_bin == "medium":
        floor += 0.04
    if truthy(row.get("border_truncated")):
        floor += 0.08
    if number(row.get("crop_mean_luma")) < 32.0:
        floor += 0.05
    return min(floor, 0.94)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--base-confidence", type=float, default=0.72)
    parser.add_argument("--minimum-track-frames", type=int, default=3)
    parser.add_argument("--minimum-unique-track-frames", type=int, default=2)
    parser.add_argument("--minimum-track-support-ratio", type=float, default=0.80)
    parser.add_argument("--maximum-supervised-frames-per-track", type=int, default=3)
    parser.add_argument("--minimum-accepted-tracks", type=int, default=100)
    parser.add_argument("--minimum-colors", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.checkpoint) < 2:
        raise ValueError("at least two checkpoints are required")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage235 evidence")
    if not 0.0 < args.minimum_track_support_ratio <= 1.0:
        raise ValueError("minimum track support ratio must be in (0, 1]")
    if args.minimum_unique_track_frames > args.minimum_track_frames:
        raise ValueError("unique frame minimum cannot exceed track frame minimum")

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    colors = json.loads(args.labels.read_text(encoding="utf-8"))["colors"]
    forbidden_colors = {"unknown", "other"}
    dataset_root = args.dataset_root.resolve()

    selected: list[int] = []
    exclusion_reasons: Counter[str] = Counter()
    for index, row in enumerate(rows):
        reasons = []
        if row.get("split") != "train":
            reasons.append("not_train")
        if row.get("source_dataset") != "NightOwls":
            reasons.append("wrong_source")
        if not truthy(row.get("crop_quality_usable")):
            reasons.append("crop_quality_unusable")
        if row.get("cross_source_duplicate_of"):
            reasons.append("cross_source_duplicate")
        if number(row.get("crop_width")) < 32 or number(row.get("crop_height")) < 24:
            reasons.append("crop_too_small_for_color")
        if number(row.get("crop_contrast")) < 10.0:
            reasons.append("contrast_too_low")
        if number(row.get("crop_mean_luma")) < 12.0:
            reasons.append("severely_underexposed")
        path_value = row.get("crop_path", "")
        if not path_value:
            reasons.append("missing_crop_path")
        else:
            path = (dataset_root / path_value).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError:
                reasons.append("crop_path_escape")
            if not path.is_file():
                reasons.append("missing_crop_file")
        if reasons:
            for reason in set(reasons):
                exclusion_reasons[reason] += 1
        else:
            selected.append(index)

    class ReviewDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, local_index: int):
            row = rows[selected[local_index]]
            path = (dataset_root / row["crop_path"]).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == "hflip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = ImageEnhance.Contrast(
                    ImageEnhance.Brightness(image).enhance(0.80)
                ).enhance(1.08)
            return self.transform(image), local_index

    device = torch.device(args.device)
    views = ("original", "hflip", "dim")
    predictions: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence: dict[str, dict[str, object]] = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint.get("colors", [])) != colors:
            raise RuntimeError(f"{name} color label order mismatch")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        predictions[name] = [[None for _ in views] for _ in selected]
        for view_index, view in enumerate(views):
            loader = DataLoader(
                ReviewDataset(transform, view),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for local, pred, conf in zip(
                        local_indexes.tolist(),
                        predicted.cpu().tolist(),
                        confidence.cpu().tolist(),
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

    frame_results: dict[int, dict[str, object]] = {}
    frame_reasons: Counter[str] = Counter()
    frame_consensus_colors: Counter[str] = Counter()
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        pairs = [predictions[name][local_index] for name, _ in args.checkpoint]
        flat = [value for pair in pairs for value in pair if value is not None]
        predicted_ids = [value[0] for value in flat]
        confidences = [value[1] for value in flat]
        view_consistent = all(
            all(value is not None for value in pair)
            and len({value[0] for value in pair if value is not None}) == 1
            for pair in pairs
        )
        teacher_consistent = bool(predicted_ids) and len(set(predicted_ids)) == 1
        consensus = colors[predicted_ids[0]] if teacher_consistent else "unknown"
        minimum_confidence = min(confidences, default=0.0)
        floor = frame_confidence_floor(row, args.base_confidence)
        passed = (
            view_consistent
            and teacher_consistent
            and consensus not in forbidden_colors
            and minimum_confidence >= floor
        )
        if not view_consistent:
            frame_reasons["view_conflict"] += 1
        if not teacher_consistent:
            frame_reasons["teacher_conflict"] += 1
        if consensus in forbidden_colors:
            frame_reasons["unknown_or_other_consensus"] += 1
        if minimum_confidence < floor:
            frame_reasons["confidence_below_adaptive_floor"] += 1
        if passed:
            frame_consensus_colors[consensus] += 1
        frame_results[row_index] = {
            "passed": passed,
            "consensus": consensus,
            "minimum_confidence": minimum_confidence,
            "floor": floor,
            "views": ";".join(
                f"{name}:" + "/".join(colors[value[0]] if value else "missing" for value in pair)
                for (name, _), pair in zip(args.checkpoint, pairs)
            ),
        }

    tracks: dict[str, list[int]] = defaultdict(list)
    for row_index in selected:
        tracks[rows[row_index].get("track_key", "")].append(row_index)

    accepted_track_labels: dict[str, str] = {}
    supervised_row_indexes: set[int] = set()
    consistency_only_indexes: set[int] = set()
    track_reasons: Counter[str] = Counter()
    accepted_track_colors: Counter[str] = Counter()
    accepted_row_colors: Counter[str] = Counter()
    accepted_size_bins: Counter[str] = Counter()
    accepted_recordings: Counter[str] = Counter()

    for track_key, row_indexes in sorted(tracks.items()):
        passed_indexes = [index for index in row_indexes if frame_results[index]["passed"]]
        class_counts = Counter(str(frame_results[index]["consensus"]) for index in passed_indexes)
        if not class_counts:
            track_reasons["no_frame_consensus"] += 1
            continue
        winning_color, winning_count = class_counts.most_common(1)[0]
        support_ratio = winning_count / len(row_indexes)
        competing_count = sum(count for color, count in class_counts.items() if color != winning_color)
        winning_indexes = [
            index for index in passed_indexes if frame_results[index]["consensus"] == winning_color
        ]
        unique_winning = [index for index in winning_indexes if not rows[index].get("internal_near_duplicate_of")]
        if len(row_indexes) < args.minimum_track_frames:
            track_reasons["track_too_short"] += 1
            continue
        if winning_count < args.minimum_track_frames:
            track_reasons["insufficient_consensus_frames"] += 1
            continue
        if len(unique_winning) < args.minimum_unique_track_frames:
            track_reasons["insufficient_non_near_duplicate_support"] += 1
            continue
        if support_ratio < args.minimum_track_support_ratio:
            track_reasons["track_support_ratio_below_floor"] += 1
            continue
        if competing_count >= 2:
            track_reasons["persistent_track_color_conflict"] += 1
            continue

        accepted_track_labels[track_key] = winning_color
        accepted_track_colors[winning_color] += 1
        accepted_recordings[rows[row_indexes[0]].get("recording_id", "unknown")] += 1
        ranked_unique = sorted(
            unique_winning,
            key=lambda index: (
                -float(frame_results[index]["minimum_confidence"]),
                -number(rows[index].get("quality_score")),
                int(number(rows[index].get("frame_index"))),
                rows[index].get("crop_path", ""),
            ),
        )
        chosen = ranked_unique[: args.maximum_supervised_frames_per_track]
        supervised_row_indexes.update(chosen)
        consistency_only_indexes.update(set(winning_indexes) - set(chosen))
        for index in chosen:
            accepted_row_colors[winning_color] += 1
            accepted_size_bins[rows[index].get("size_bin", "unknown")] += 1

    review_fields = [
        "image_path",
        "color_teacher_consensus",
        "color_teacher_class",
        "color_teacher_confidence",
        "color_teacher_required_confidence",
        "color_teacher_views",
        "color_track_consensus",
        "color_track_label",
        "color_track_policy",
        "review_status",
        "color_review_status",
        "label_confidence",
        "review_method",
        "review_score",
    ]
    for index, row in enumerate(rows):
        result = frame_results.get(index)
        row["image_path"] = row.get("crop_path", row.get("image_path", ""))
        row["color"] = "unknown"
        row["color_supervised"] = "false"
        row["formal_train_eligible"] = "false"
        row["review_status"] = "rejected"
        row["color_review_status"] = "not_audited"
        row["label_confidence"] = "unknown"
        row["review_method"] = "stage235_nightowls_dual_teacher_three_view_track_consensus"
        row["review_score"] = "0.000000"
        if result is None:
            row["color_teacher_consensus"] = "not_audited"
            row["color_teacher_class"] = "unknown"
            row["color_teacher_confidence"] = "0.000000"
            row["color_teacher_required_confidence"] = "0.000000"
            row["color_teacher_views"] = "not_audited"
        else:
            row["color_teacher_consensus"] = "frame_accepted" if result["passed"] else "frame_rejected"
            row["color_teacher_class"] = str(result["consensus"])
            row["color_teacher_confidence"] = f"{float(result['minimum_confidence']):.6f}"
            row["color_teacher_required_confidence"] = f"{float(result['floor']):.6f}"
            row["color_teacher_views"] = str(result["views"])
        track_key = row.get("track_key", "")
        track_label = accepted_track_labels.get(track_key, "unknown")
        row["color_track_label"] = track_label
        row["color_track_policy"] = (
            "minimum three consensus frames, minimum two non-near-duplicate supports, "
            "support ratio >= configured floor, fewer than two competing frames, "
            "maximum three supervised frames per track"
        )
        if index in supervised_row_indexes:
            row["color"] = track_label
            row["color_supervised"] = "true"
            row["formal_train_eligible"] = "true"
            row["review_status"] = "approved"
            row["color_review_status"] = "auto_multiteacher_three_view_track_consensus"
            row["label_confidence"] = "high"
            row["review_score"] = row["color_teacher_confidence"]
            row["color_teacher_consensus"] = "accepted"
            row["color_track_consensus"] = "accepted_supervised_representative"
        elif index in consistency_only_indexes:
            row["color_track_consensus"] = "accepted_consistency_only"
        elif track_key in accepted_track_labels:
            row["color_track_consensus"] = "accepted_track_frame_not_consensus"
        else:
            row["color_track_consensus"] = "rejected_or_insufficient_track"

    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    status = (
        "complete_pending_agent_visual_audit"
        if len(accepted_track_labels) >= args.minimum_accepted_tracks
        and len(accepted_track_colors) >= args.minimum_colors
        else "complete_fail_closed_insufficient_track_consensus"
    )
    report = {
        "schema_version": "stage235-nightowls-color-track-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_rows": len(rows),
        "audited_rows": len(selected),
        "excluded_rows": len(rows) - len(selected),
        "exclusion_reasons_nonexclusive": dict(sorted(exclusion_reasons.items())),
        "frame_consensus_rows": sum(frame_consensus_colors.values()),
        "frame_consensus_color_counts": dict(sorted(frame_consensus_colors.items())),
        "frame_rejection_reasons_nonexclusive": dict(sorted(frame_reasons.items())),
        "tracks_audited": len(tracks),
        "accepted_tracks": len(accepted_track_labels),
        "accepted_track_color_counts": dict(sorted(accepted_track_colors.items())),
        "track_rejection_reasons": dict(sorted(track_reasons.items())),
        "supervised_representative_rows": len(supervised_row_indexes),
        "supervised_row_color_counts": dict(sorted(accepted_row_colors.items())),
        "supervised_size_bin_counts": dict(sorted(accepted_size_bins.items())),
        "accepted_recording_count": len(accepted_recordings),
        "accepted_recording_track_counts": dict(sorted(accepted_recordings.items())),
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "checkpoints": checkpoint_evidence,
        "thresholds": {
            "base_frame_confidence": args.base_confidence,
            "small_addition": 0.10,
            "medium_addition": 0.04,
            "border_truncated_addition": 0.08,
            "luma_below_32_addition": 0.05,
            "maximum_frame_confidence_floor": 0.94,
            "minimum_track_frames": args.minimum_track_frames,
            "minimum_unique_track_frames": args.minimum_unique_track_frames,
            "minimum_track_support_ratio": args.minimum_track_support_ratio,
            "maximum_supervised_frames_per_track": args.maximum_supervised_frames_per_track,
            "persistent_conflict_rejection_at_frames": 2,
        },
        "minimum_stage_acceptance": {
            "accepted_tracks": args.minimum_accepted_tracks,
            "distinct_colors": args.minimum_colors,
        },
        "views": ["original", "horizontal_flip", "brightness_0.80_contrast_1.08"],
        "policy": {
            "official_train_pixels_only": True,
            "teacher_predictions_create_research_only_pseudo_labels": True,
            "track_grouping_required": True,
            "internal_near_duplicates_not_independent_support": True,
            "maximum_three_supervised_frames_per_track": True,
            "ambiguous_frames_changed_to_unknown_unsupervised": True,
            "research_only": True,
            "deployment_eligible": False,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": (
            "accepted representatives require agent visual audit before any isolated training; "
            "this stage alone does not authorize training or deployment"
        ),
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "complete_pending_agent_visual_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
