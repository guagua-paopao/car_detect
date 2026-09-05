#!/usr/bin/env python3
"""Apply a validation-frozen color consensus rule to train-only UA tracks.

The validation report supplies only fixed acceptance parameters.  Train images
are never compared with validation truth.  Two teachers must agree after
deterministic-view aggregation, foreground pixels may only accept or reject
that decision, and a track needs repeated conflict-free frame decisions.
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
from typing import Any, Callable


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from scripts.build_bmd45_track_color_pseudolabels import (  # noqa: E402
    evenly_spaced,
    foreground_color_evidence,
)
from scripts.evaluate_stage73_color_pseudolabel_rule import (  # noqa: E402
    aggregate_teachers,
    aggregate_views,
    foreground_compatible,
    passes_gate,
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


def assert_non_test_asset(path: Path, label: str) -> Path:
    resolved = path.resolve()
    parts = {part.lower() for part in resolved.parts}
    lowered = str(resolved).lower()
    if {"test", "testing"} & parts or "vcas_rtsp_demo_60s" in lowered:
        raise RuntimeError(f"{label} references test or frozen-video evidence: {resolved}")
    return resolved


def require_manifest_dataset_root(manifest: Path, dataset_root: Path) -> None:
    expected = manifest.parent.resolve()
    if dataset_root.resolve() != expected:
        raise RuntimeError(
            f"dataset root must equal the train manifest directory: expected {expected}, "
            f"got {dataset_root.resolve()}"
        )


def load_color_contract(path: Path, expected_sha256: str) -> tuple[list[str], str]:
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("color label contract SHA256 mismatch")
    colors = json.loads(path.read_text(encoding="utf-8"))["colors"]
    if not isinstance(colors, list) or len(colors) != len(set(colors)) or "unknown" not in colors:
        raise RuntimeError("invalid ordered color contract")
    return list(colors), actual


def load_validation_rule(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("validation rule report SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass_validation_rule_available":
        raise RuntimeError("validation rule report did not pass")
    policy = payload.get("policy", {})
    if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
        raise RuntimeError("validation rule report is contaminated by test or frozen-video access")
    if policy.get("usage") != "validation_threshold_selection_only":
        raise RuntimeError("validation rule report has an unexpected usage contract")
    if policy.get("truth_used_as_teacher_input") is not False:
        raise RuntimeError("validation truth influenced teacher predictions")
    if policy.get("foreground_pixels_only_accept_or_reject_teacher_consensus") is not True:
        raise RuntimeError("validation rule permits foreground evidence to invent labels")
    selected = payload.get("selected")
    if not isinstance(selected, dict) or not passes_gate(selected):
        raise RuntimeError("validation-selected rule no longer satisfies its statistical gates")
    parameters = selected.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("validation-selected rule lacks parameters")
    required = {
        "minimum_views",
        "confidence_floor",
        "foreground_mode",
        "chromatic_floor",
        "achromatic_floor",
    }
    if set(parameters) != required:
        raise RuntimeError("validation-selected rule parameter schema mismatch")
    if int(parameters["minimum_views"]) not in {2, 3}:
        raise RuntimeError("invalid validation-selected minimum view count")
    if str(parameters["foreground_mode"]) not in {"exact_strong", "group_compatible"}:
        raise RuntimeError("invalid validation-selected foreground mode")
    return dict(parameters), actual


def choose_track_decision(
    decisions: list[str], minimum_frames: int, minimum_agreement: float
) -> tuple[str, float, str]:
    known = [label for label in decisions if label != "unknown"]
    if len(known) < minimum_frames:
        return "unknown", 0.0, "insufficient_accepted_frames"
    label, count = Counter(known).most_common(1)[0]
    if any(candidate != label for candidate in known):
        return "unknown", count / len(decisions), "accepted_frame_conflict"
    share = count / len(decisions)
    if share < minimum_agreement:
        return "unknown", share, "insufficient_track_agreement"
    return label, share, "accepted"


def foreground_support(evidence: dict[str, Any], predicted: str) -> float:
    scores = evidence.get("scores")
    if isinstance(scores, dict):
        return float(scores.get(predicted, 0.0) or 0.0)
    return float(evidence.get("score", 0.0) or 0.0)


def select_readable_audit_rows(
    track_rows: list[dict[str, str]],
    audit_frames: int,
    minimum_frames: int,
    is_readable: Callable[[dict[str, str]], bool],
) -> tuple[list[dict[str, str]], int]:
    ordered = sorted(track_rows, key=lambda row: int(row["frame_number"]))
    # Probe a wider, evenly distributed set so a few stale crop paths do not
    # discard an otherwise usable track.  Unreadable candidates are never
    # passed to foreground extraction or a DataLoader.
    probe_count = min(len(ordered), max(audit_frames * 4, minimum_frames))
    probes = evenly_spaced(ordered, probe_count)
    readable = [row for row in probes if is_readable(row)]
    rejected = len(probes) - len(readable)
    if len(readable) < minimum_frames:
        return [], rejected
    return evenly_spaced(readable, audit_frames), rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--validation-rule-report", type=Path, required=True)
    parser.add_argument("--expected-validation-rule-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-track-frames", type=int, default=3)
    parser.add_argument("--audit-frames-per-track", type=int, default=5)
    parser.add_argument("--track-decision-agreement", type=float, default=0.60)
    parser.add_argument("--output-frames-per-track", type=int, default=3)
    parser.add_argument("--maximum-per-class", type=int, default=6000)
    parser.add_argument("--minimum-total", type=int, default=500)
    parser.add_argument("--minimum-classes", type=int, default=5)
    parser.add_argument("--minimum-rows-per-counted-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two independently trained checkpoints are required")
    if args.audit_frames_per_track < args.minimum_track_frames:
        raise ValueError("audit frame count must cover the track minimum")
    if not 0.0 < args.track_decision_agreement <= 1.0:
        raise ValueError("track decision agreement must be in (0, 1]")
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    manifest = assert_non_test_asset(args.manifest, "train manifest")
    dataset_root = assert_non_test_asset(args.dataset_root, "dataset root")
    require_manifest_dataset_root(manifest, dataset_root)
    labels_path = assert_non_test_asset(args.labels, "label contract")
    rule_path = assert_non_test_asset(args.validation_rule_report, "validation rule report")
    colors, labels_sha = load_color_contract(labels_path, args.expected_labels_sha256)
    rule, rule_sha = load_validation_rule(rule_path, args.expected_validation_rule_sha256)

    import torch
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        source_rows = list(reader)
    if any(row.get("split") not in {"train", "validation"} for row in source_rows):
        raise RuntimeError("source manifest contains a non-training-partition split")
    if any(truthy(row.get("color_supervised")) or row.get("color") != "unknown" for row in source_rows):
        raise RuntimeError("source manifest unexpectedly contains color supervision")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row.get("split") != "train":
            continue
        if truthy(row.get("occluded")) or truthy(row.get("truncated")):
            continue
        if row.get("crop_quality") != "good":
            continue
        track = str(row.get("track_group", "")).strip()
        if track:
            grouped[track].append(row)
    image_filter_reasons = Counter()

    def is_readable(row: dict[str, str]) -> bool:
        path = (dataset_root / row["image_path"]).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError:
            image_filter_reasons["outside_dataset_root"] += 1
            return False
        if not path.is_file():
            image_filter_reasons["missing"] += 1
            return False
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (OSError, ValueError):
            image_filter_reasons["unreadable"] += 1
            return False
        image_filter_reasons["readable"] += 1
        return True

    selected_by_track: dict[str, list[dict[str, str]]] = {}
    for track, track_rows in grouped.items():
        if len(track_rows) < args.minimum_track_frames:
            continue
        selected, _ = select_readable_audit_rows(
            track_rows,
            args.audit_frames_per_track,
            args.minimum_track_frames,
            is_readable,
        )
        if selected:
            selected_by_track[track] = selected
    audit_rows: list[tuple[str, dict[str, str]]] = [
        (track, row)
        for track, rows in sorted(selected_by_track.items())
        for row in rows
    ]
    if not audit_rows:
        raise RuntimeError("no eligible train-only track frames")

    foreground: list[dict[str, Any]] = []
    foreground_reasons = Counter()
    for _, row in audit_rows:
        path = (dataset_root / row["image_path"]).resolve()
        path.relative_to(dataset_root)
        evidence = foreground_color_evidence(path)
        foreground.append(evidence)
        foreground_reasons[str(evidence.get("reason", "unknown"))] += 1

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
        predictions[name] = [[] for _ in audit_rows]
        for view in views:
            loader = DataLoader(
                AuditDataset(transform, view),
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

    frame_decisions: dict[str, list[tuple[dict[str, str], str, float, float]]] = defaultdict(list)
    frame_rejections = Counter()
    minimum_views = int(rule["minimum_views"])
    confidence_floor = float(rule["confidence_floor"])
    foreground_mode = str(rule["foreground_mode"])
    chromatic_floor = float(rule["chromatic_floor"])
    achromatic_floor = float(rule["achromatic_floor"])
    for index, (track, row) in enumerate(audit_rows):
        teacher_views = {name: predictions[name][index] for name, _ in args.checkpoint}
        teacher_decisions = {
            name: aggregate_views(items, minimum_views, confidence_floor)
            for name, items in teacher_views.items()
        }
        if any(decision is None for decision in teacher_decisions.values()):
            frame_rejections["teacher_view_or_confidence_reject"] += 1
            frame_decisions[track].append((row, "unknown", 0.0, 0.0))
            continue
        decision = aggregate_teachers(teacher_views, minimum_views, confidence_floor)
        if decision is None:
            frame_rejections["teacher_label_conflict"] += 1
            frame_decisions[track].append((row, "unknown", 0.0, 0.0))
            continue
        predicted, confidence = decision
        evidence = foreground[index]
        if not foreground_compatible(
            evidence,
            predicted,
            foreground_mode,
            chromatic_floor,
            achromatic_floor,
        ):
            frame_rejections["foreground_incompatible"] += 1
            frame_decisions[track].append((row, "unknown", 0.0, 0.0))
            continue
        frame_rejections["accepted"] += 1
        frame_decisions[track].append(
            (row, predicted, float(confidence), foreground_support(evidence, predicted))
        )

    accepted_by_class: dict[str, list[tuple[str, dict[str, str], float, float, float]]] = defaultdict(list)
    track_reasons = Counter()
    for track, items in sorted(frame_decisions.items()):
        decisions = [item[1] for item in items]
        label, share, reason = choose_track_decision(
            decisions, args.minimum_track_frames, args.track_decision_agreement
        )
        track_reasons[reason] += 1
        if label == "unknown":
            continue
        passing = [item for item in items if item[1] == label]
        passing.sort(
            key=lambda item: (
                item[2] + item[3],
                float(item[0].get("edge_variance", 0.0) or 0.0),
            ),
            reverse=True,
        )
        accepted_by_class[label].extend(
            (track, item[0], item[2], item[3], share)
            for item in passing[: args.output_frames_per_track]
        )

    accepted: list[tuple[str, str, dict[str, str], float, float, float]] = []
    for color, items in sorted(accepted_by_class.items()):
        items.sort(
            key=lambda item: (
                -item[2],
                -item[3],
                item[0],
                int(item[1]["frame_number"]),
            )
        )
        accepted.extend((color, *item) for item in items[: args.maximum_per_class])

    extra_fields = [
        "color_review_status",
        "review_score",
        "review_model",
        "formal_train_eligible",
        "stage73_foreground_score",
        "stage73_track_agreement",
        "stage73_teacher_confidence",
        "stage73_validation_rule_sha256",
    ]
    output_rows: list[dict[str, str]] = []
    for color, track, source, confidence, support, share in accepted:
        row = dict(source)
        row.update({
            "color": color,
            "color_supervised": "true",
            "review_status": "approved",
            "annotation_source": "uadetrac_validation_frozen_temporal_multiteacher_consensus_v2",
            "review_method": "validation-frozen two-teacher view consensus+foreground compatibility+conflict-free track repetition",
            "color_review_status": "auto_validation_frozen_track_consensus",
            "review_score": f"{min(confidence, support):.6f}",
            "review_model": ";".join(name for name, _ in args.checkpoint),
            "formal_train_eligible": "false",
            "license_train_eligible": "false",
            "pseudo_label": "true",
            "pseudo_label_confidence": f"{confidence:.6f}",
            "stage73_foreground_score": f"{support:.6f}",
            "stage73_track_agreement": f"{share:.6f}",
            "stage73_teacher_confidence": f"{confidence:.6f}",
            "stage73_validation_rule_sha256": rule_sha,
        })
        output_rows.append(row)

    output_fields = fields + [field for field in extra_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    accepted_counts = Counter(row["color"] for row in output_rows)
    counted_classes = sum(
        count >= args.minimum_rows_per_counted_class for count in accepted_counts.values()
    )
    status = (
        "pass_train_only_manifest_available"
        if len(output_rows) >= args.minimum_total and counted_classes >= args.minimum_classes
        else "fail_closed_insufficient_train_only_evidence"
    )
    report = {
        "schema_version": "stage73-uadetrac-color-pseudolabel-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "eligibility": "research-only_non-deployable",
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256(manifest),
        "validation_rule_report": str(rule_path),
        "validation_rule_report_sha256": rule_sha,
        "validation_frozen_parameters": rule,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "eligible_train_tracks": len(selected_by_track),
        "audited_train_frames": len(audit_rows),
        "image_filter_reasons": dict(sorted(image_filter_reasons.items())),
        "foreground_reasons": dict(sorted(foreground_reasons.items())),
        "frame_decisions": dict(sorted(frame_rejections.items())),
        "track_decisions": dict(sorted(track_reasons.items())),
        "accepted_tracks": len({row["track_group"] for row in output_rows}),
        "accepted_rows": len(output_rows),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "classes_with_minimum_rows": counted_classes,
        "checkpoints": checkpoint_evidence,
        "track_parameters": {
            "minimum_track_frames": args.minimum_track_frames,
            "audit_frames_per_track": args.audit_frames_per_track,
            "track_decision_agreement": args.track_decision_agreement,
            "output_frames_per_track": args.output_frames_per_track,
            "maximum_per_class": args.maximum_per_class,
            "minimum_total": args.minimum_total,
            "minimum_classes": args.minimum_classes,
            "minimum_rows_per_counted_class": args.minimum_rows_per_counted_class,
        },
        "policy": {
            "validation_report_used_for_fixed_parameters_only": True,
            "validation_images_or_truth_used_for_train_inference": False,
            "teachers_create_consensus": True,
            "foreground_pixels_only_accept_or_reject": True,
            "conflicting_accepted_track_frames_rejected": True,
            "train_split_images_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "license_train_eligible": False,
        },
        "decision": "only pass permits isolated research-only candidate training; failure imports zero rows",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "accepted_rows": len(output_rows), "counts": accepted_counts}))
    return 0 if status.startswith("pass_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
