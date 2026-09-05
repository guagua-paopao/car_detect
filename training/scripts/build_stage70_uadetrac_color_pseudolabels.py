#!/usr/bin/env python3
"""Create conservative train-only color pseudo labels for Stage70 UA tracks.

Foreground pixels create a proposal.  Multiple independently trained models,
three deterministic views, and multiple frames from the same official track
may only accept or reject that proposal; they never invent or replace it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from scripts.build_bmd45_track_color_pseudolabels import (  # noqa: E402
    evenly_spaced,
    foreground_color_evidence,
)
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


def load_color_contract(path: Path, expected_sha256: str) -> tuple[list[str], str]:
    actual_sha256 = sha256(path)
    if actual_sha256.lower() != expected_sha256.strip().lower():
        raise RuntimeError(
            f"color label contract SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    colors = payload.get("colors")
    if not isinstance(colors, list) or not colors or not all(isinstance(item, str) for item in colors):
        raise RuntimeError("color label contract is missing an ordered colors list")
    if len(colors) != len(set(colors)) or "unknown" not in colors:
        raise RuntimeError("color label contract contains duplicates or lacks unknown")
    return colors, actual_sha256


def validate_checkpoint_color_contract(name: str, checkpoint_colors: object, colors: list[str]) -> None:
    if list(checkpoint_colors or []) != colors:
        raise RuntimeError(f"{name} color label order mismatch")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=/absolute/path.pt")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return name, path


def assert_train_asset(path: Path, label: str) -> Path:
    resolved = path.resolve()
    parts = {part.lower() for part in resolved.parts}
    lowered = str(resolved).lower()
    if {"test", "testing"} & parts or "vcas_rtsp_demo_60s" in lowered:
        raise RuntimeError(f"{label} is not train-only: {resolved}")
    return resolved


def choose_track_proposal(
    evidences: list[dict[str, object]], minimum_frames: int, minimum_agreement: float
) -> tuple[str, float, str]:
    labels = [str(item.get("label", "unknown")) for item in evidences]
    known = [label for label in labels if label != "unknown"]
    if len(known) < minimum_frames:
        return "unknown", 0.0, "insufficient_foreground_frames"
    label, count = Counter(known).most_common(1)[0]
    # Unknown evidence remains in the denominator: abstention cannot make a
    # weak proposal look artificially consistent.
    share = count / len(labels)
    if share < minimum_agreement:
        return "unknown", share, "foreground_track_conflict"
    return label, share, "accepted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--teacher-confidence", type=float, default=0.75)
    parser.add_argument("--track-evidence-agreement", type=float, default=0.80)
    parser.add_argument("--minimum-evidence-frames", type=int, default=3)
    parser.add_argument("--audit-frames-per-track", type=int, default=5)
    parser.add_argument("--output-frames-per-track", type=int, default=3)
    parser.add_argument("--maximum-per-class", type=int, default=6000)
    parser.add_argument("--minimum-total", type=int, default=500)
    parser.add_argument("--minimum-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two independently trained checkpoints are required")
    if args.audit_frames_per_track < args.minimum_evidence_frames:
        raise ValueError("audit frame count must cover the evidence minimum")
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    manifest = assert_train_asset(args.manifest, "manifest")
    dataset_root = assert_train_asset(args.dataset_root, "dataset root")

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if any(row.get("split") not in {"train", "validation"} for row in rows):
        raise RuntimeError("manifest contains a non-training-partition split")
    if any(truthy(row.get("color_supervised")) or row.get("color") != "unknown" for row in rows):
        raise RuntimeError("source manifest unexpectedly contains color supervision")
    colors, labels_sha256 = load_color_contract(args.labels, args.expected_labels_sha256)
    allowed_proposals = set(colors) - {"unknown"}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            continue
        if truthy(row.get("occluded")) or truthy(row.get("truncated")):
            continue
        if row.get("crop_quality") != "good":
            continue
        track = str(row.get("track_group", "")).strip()
        if track:
            grouped[track].append(row)
    selected_by_track = {
        track: evenly_spaced(sorted(track_rows, key=lambda row: int(row["frame_number"])), args.audit_frames_per_track)
        for track, track_rows in grouped.items()
        if len(track_rows) >= args.minimum_evidence_frames
    }

    foreground_by_key: dict[tuple[str, str], dict[str, object]] = {}
    foreground_reasons = Counter()
    proposal_counts = Counter()
    proposal_by_track: dict[str, tuple[str, float]] = {}
    audit_rows: list[tuple[str, dict[str, str]]] = []
    for track, track_rows in sorted(selected_by_track.items()):
        evidences: list[dict[str, object]] = []
        for row in track_rows:
            path = (dataset_root / row["image_path"]).resolve()
            path.relative_to(dataset_root)
            evidence = foreground_color_evidence(path)
            foreground_by_key[(track, row["image_path"])] = evidence
            evidences.append(evidence)
            foreground_reasons[str(evidence.get("reason", "unknown"))] += 1
        proposal, share, reason = choose_track_proposal(
            evidences, args.minimum_evidence_frames, args.track_evidence_agreement
        )
        foreground_reasons[f"track_{reason}"] += 1
        if proposal == "unknown" or proposal not in allowed_proposals:
            continue
        proposal_by_track[track] = (proposal, share)
        proposal_counts[proposal] += 1
        audit_rows.extend((track, row) for row in track_rows)

    if not audit_rows:
        raise RuntimeError("no foreground-consistent train tracks are available for teacher audit")

    class AuditDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(audit_rows)

        def __getitem__(self, index: int):
            _, row = audit_rows[index]
            path = (dataset_root / row["image_path"]).resolve()
            path.relative_to(dataset_root)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self.view == "hflip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.80)).enhance(1.08)
            return self.transform(image), index

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    views = ("original", "hflip", "dim")
    predictions: dict[str, list[list[tuple[int, float] | None]]] = {}
    checkpoint_evidence: dict[str, dict[str, object]] = {}
    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        validate_checkpoint_color_contract(name, checkpoint.get("colors", []), colors)
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        predictions[name] = [[None for _ in views] for _ in audit_rows]
        for view_index, view in enumerate(views):
            loader = DataLoader(
                AuditDataset(transform, view), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, indexes in loader:
                    _, logits = model(images.to(device, non_blocking=True))
                    confidence, predicted = logits.softmax(dim=1).max(dim=1)
                    for index, pred, conf in zip(indexes.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                        predictions[name][index][view_index] = (int(pred), float(conf))
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"), "input_size": input_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    passed_by_track: dict[str, list[tuple[dict[str, str], float, float]]] = defaultdict(list)
    teacher_reasons = Counter()
    for index, (track, row) in enumerate(audit_rows):
        proposal = proposal_by_track[track][0]
        teacher_views = [predictions[name][index] for name, _ in args.checkpoint]
        flat = [item for group in teacher_views for item in group if item is not None]
        ids = [item[0] for item in flat]
        confidences = [item[1] for item in flat]
        view_consistent = all(
            all(item is not None for item in group)
            and len({item[0] for item in group if item is not None}) == 1
            for group in teacher_views
        )
        teacher_consistent = bool(ids) and len(set(ids)) == 1
        predicted = colors[ids[0]] if teacher_consistent else "unknown"
        confidence = min(confidences, default=0.0)
        if not view_consistent:
            teacher_reasons["view_conflict"] += 1
        if not teacher_consistent:
            teacher_reasons["teacher_conflict"] += 1
        if teacher_consistent and predicted != proposal:
            teacher_reasons["teacher_foreground_disagreement"] += 1
        if confidence < args.teacher_confidence:
            teacher_reasons["confidence_below_floor"] += 1
        if view_consistent and teacher_consistent and predicted == proposal and confidence >= args.teacher_confidence:
            foreground = foreground_by_key[(track, row["image_path"])]
            passed_by_track[track].append((row, confidence, float(foreground.get("score", 0.0))))

    accepted_by_class: dict[str, list[tuple[str, dict[str, str], float, float]]] = defaultdict(list)
    rejected_tracks = Counter()
    for track, (proposal, _) in sorted(proposal_by_track.items()):
        passed = passed_by_track.get(track, [])
        if len(passed) < args.minimum_evidence_frames:
            rejected_tracks["insufficient_teacher_approved_frames"] += 1
            continue
        passed.sort(key=lambda item: (item[1] + item[2], float(item[0]["edge_variance"])), reverse=True)
        accepted_by_class[proposal].extend((track, *item) for item in passed[:args.output_frames_per_track])

    accepted: list[tuple[str, str, dict[str, str], float, float]] = []
    for color, items in sorted(accepted_by_class.items()):
        items.sort(key=lambda item: (-item[2], -item[3], item[0], int(item[1]["frame_number"])))
        accepted.extend((color, *item) for item in items[:args.maximum_per_class])

    extra_fields = [
        "color_review_status", "review_score", "review_model", "formal_train_eligible",
        "stage70_foreground_score", "stage70_track_agreement", "stage70_teacher_confidence",
    ]
    output_rows: list[dict[str, str]] = []
    for color, track, source, teacher_confidence, foreground_score in accepted:
        row = dict(source)
        row.update({
            "color": color, "color_supervised": "true", "review_status": "approved",
            "annotation_source": "uadetrac_foreground_temporal_multiteacher_consensus_v1",
            "review_method": "foreground_proposal+official_track_temporal_consensus+multiteacher_three_view_acceptance",
            "color_review_status": "auto_multiframe_multiteacher_three_view_consensus",
            "review_score": f"{min(teacher_confidence, foreground_score):.6f}",
            "review_model": ";".join(name for name, _ in args.checkpoint),
            "formal_train_eligible": "false", "license_train_eligible": "false",
            "pseudo_label": "true", "pseudo_label_confidence": f"{teacher_confidence:.6f}",
            "stage70_foreground_score": f"{foreground_score:.6f}",
            "stage70_track_agreement": f"{proposal_by_track[track][1]:.6f}",
            "stage70_teacher_confidence": f"{teacher_confidence:.6f}",
        })
        output_rows.append(row)
    output_fields = fields + [field for field in extra_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output_rows)
    accepted_counts = Counter(row["color"] for row in output_rows)
    status = "pass" if len(output_rows) >= args.minimum_total and len(accepted_counts) >= args.minimum_classes else "fail"
    report = {
        "schema_version": "stage70-uadetrac-color-pseudolabel-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "eligibility": "research-only_non-deployable",
        "color_label_contract": {
            "path": str(args.labels.resolve()), "sha256": labels_sha256, "colors": colors,
        },
        "input_manifest": str(manifest), "input_manifest_sha256": sha256(manifest),
        "output_manifest": str(args.output_manifest.resolve()), "output_manifest_sha256": sha256(args.output_manifest),
        "train_tracks_with_visible_good_frames": len(grouped),
        "foreground_audited_tracks": len(selected_by_track),
        "foreground_proposal_tracks": len(proposal_by_track),
        "foreground_proposal_counts": dict(sorted(proposal_counts.items())),
        "foreground_reasons_nonexclusive": dict(sorted(foreground_reasons.items())),
        "teacher_audited_frames": len(audit_rows),
        "teacher_rejection_reasons_nonexclusive": dict(sorted(teacher_reasons.items())),
        "teacher_rejected_tracks": dict(sorted(rejected_tracks.items())),
        "accepted_tracks": len({row["track_group"] for row in output_rows}),
        "accepted_rows": len(output_rows), "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "checkpoints": checkpoint_evidence,
        "parameters": {
            "teacher_confidence": args.teacher_confidence,
            "track_evidence_agreement": args.track_evidence_agreement,
            "minimum_evidence_frames": args.minimum_evidence_frames,
            "audit_frames_per_track": args.audit_frames_per_track,
            "output_frames_per_track": args.output_frames_per_track,
            "maximum_per_class": args.maximum_per_class,
            "minimum_total": args.minimum_total, "minimum_classes": args.minimum_classes,
        },
        "policy": {
            "foreground_pixels_create_proposal": True, "teachers_only_accept_or_reject": True,
            "model_predictions_replace_proposals": False, "multi_frame_consistency_required": True,
            "all_teachers_and_original_flip_dim_views_must_agree": True,
            "train_split_only": True, "validation_or_test_used": False,
            "frozen_video_used": False, "production_model_modified": False,
            "deployment_performed": False, "license_train_eligible": False,
        },
        "decision": "pass permits research-only Stage70 training; fail keeps every source color unknown",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
