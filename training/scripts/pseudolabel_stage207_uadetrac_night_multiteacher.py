#!/usr/bin/env python3
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

VIEWS = ("original", "horizontal_flip", "dim", "brighten")
COARSE_COMPATIBILITY = {
    "car": {"sedan", "suv", "mpv", "van", "pickup"},
    "bus": {"bus"},
    "van": {"van"},
    "others": set(),
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use NAME=/absolute/path")
    name, raw = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("empty checkpoint name")
    return name, Path(raw)


def parse_named_hash(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use NAME=SHA256")
    name, digest = value.split("=", 1)
    if not name or len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise argparse.ArgumentTypeError("invalid named SHA256")
    return name, digest.lower()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def canonical_label(local_labels: list[str], predicted_id: int, canonical_labels: list[str]) -> str:
    if not 0 <= predicted_id < len(local_labels):
        return "unknown"
    label = str(local_labels[predicted_id])
    return label if label in canonical_labels else "unknown"


def coarse_compatible(source_type: str, label: str) -> bool:
    return label in COARSE_COMPATIBILITY.get(source_type.strip().lower(), set())


def qualify_track(
    strong_labels: list[str], total_rows: int, minimum_frames: int, minimum_fraction: float
) -> str | None:
    if total_rows <= 0 or len(strong_labels) < minimum_frames:
        return None
    if len(set(strong_labels)) != 1:
        return None
    if len(strong_labels) / total_rows < minimum_fraction:
        return None
    return strong_labels[0] if strong_labels[0] != "unknown" else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict Stage207 UA-DETRAC night multi-teacher track audit")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_named_path)
    parser.add_argument("--expected-checkpoint-sha256", action="append", required=True, type=parse_named_hash)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--minimum-track-frames", type=int, default=2)
    parser.add_argument("--minimum-track-strong-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-authorized-rows", type=int, default=100)
    parser.add_argument("--minimum-authorized-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = (args.manifest, args.images_root, args.labels, args.output_root)
    for path in paths:
        lowered = str(path.resolve()).lower()
        if any(marker in lowered for marker in FROZEN_MARKERS) or "/test" in lowered or "\\test" in lowered:
            raise RuntimeError(f"prohibited path marker: {path}")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if len(args.checkpoint) < 5:
        raise RuntimeError("five teachers are required")
    expected_hashes = dict(args.expected_checkpoint_sha256)
    checkpoint_names = [name for name, _ in args.checkpoint]
    if len(checkpoint_names) != len(set(checkpoint_names)) or set(checkpoint_names) != set(expected_hashes):
        raise RuntimeError("checkpoint and SHA256 names must match exactly")
    if sha256_path(args.manifest).lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError("manifest SHA256 mismatch")
    if sha256_path(args.labels).lower() != args.expected_labels_sha256.lower():
        raise RuntimeError("labels SHA256 mismatch")
    if not 0 < args.confidence <= 1 or not 0 < args.minimum_track_strong_fraction <= 1:
        raise RuntimeError("invalid confidence or track fraction")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF
    from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    canonical = list(json.loads(args.labels.read_text(encoding="utf-8"))["body_types"])
    if "unknown" not in canonical:
        raise RuntimeError("canonical body taxonomy lacks unknown")
    images_root = args.images_root.resolve()
    if not images_root.is_dir() or images_root.is_symlink():
        raise RuntimeError("images root missing or symlinked")

    selected: list[int] = []
    resolved_paths: list[Path] = []
    isolated_validation_rows = 0
    for index, row in enumerate(rows):
        split = row.get("split", "").strip().lower()
        if split == "validation":
            isolated_validation_rows += 1
            continue
        if split != "train":
            raise RuntimeError(f"unexpected split at row {index + 2}: {split}")
        if row.get("body_type", "").strip().lower() != "unknown" or truthy(row.get("body_type_supervised")):
            raise RuntimeError(f"body contract violation at row {index + 2}")
        if row.get("color", "").strip().lower() != "unknown" or truthy(row.get("color_supervised")):
            raise RuntimeError(f"color contract violation at row {index + 2}")
        if row.get("stage206_scene_label", "").strip().lower() != "night":
            raise RuntimeError(f"night scene contract violation at row {index + 2}")
        relative = Path(row.get("image_path", ""))
        path = (images_root / relative).resolve()
        path.relative_to(images_root)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256_path(path) != row.get("sha256", "").strip().lower():
            raise RuntimeError(f"image SHA256 mismatch at row {index + 2}")
        selected.append(index)
        resolved_paths.append(path)
    if not selected:
        raise RuntimeError("no train rows selected")

    class ReviewDataset(Dataset):
        def __init__(self, transform, view: str) -> None:
            self.transform = transform
            self.view = view

        def __len__(self) -> int:
            return len(selected)

        def __getitem__(self, local_index: int):
            with Image.open(resolved_paths[local_index]) as opened:
                image = opened.convert("RGB")
            if self.view == "horizontal_flip":
                image = TF.hflip(image)
            elif self.view == "dim":
                image = TF.adjust_contrast(TF.adjust_brightness(image, 0.72), 1.08)
            elif self.view == "brighten":
                image = TF.adjust_contrast(TF.adjust_brightness(image, 1.22), 0.96)
            return self.transform(image), local_index

    device = torch.device(args.device)
    predictions: dict[str, list[list[tuple[str, float] | None]]] = {}
    checkpoint_evidence: dict[str, dict[str, object]] = {}
    architecture_families: set[str] = set()
    for name, checkpoint_path in args.checkpoint:
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise FileNotFoundError(checkpoint_path)
        actual_hash = sha256_path(checkpoint_path)
        if actual_hash != expected_hashes[name]:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        local_labels = list(checkpoint.get("body_types", []))
        if not local_labels or "unknown" not in local_labels or not set(local_labels) <= set(canonical):
            raise RuntimeError(f"teacher taxonomy mismatch: {name}")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        resize_mode = str(checkpoint.get("resize_mode", "stretch"))
        if resize_mode != "stretch":
            raise RuntimeError(f"unsupported teacher geometry: {name}={resize_mode}")
        architecture = str(checkpoint.get("architecture", "unknown"))
        architecture_families.add(architecture)
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        predictions[name] = [[None for _ in VIEWS] for _ in selected]
        for view_index, view in enumerate(VIEWS):
            loader = DataLoader(
                ReviewDataset(transform, view), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for images, local_indices in loader:
                    body_logits, _ = model(images.to(device, non_blocking=True))
                    confidence, predicted = body_logits.softmax(dim=1).max(dim=1)
                    for local, predicted_id, score in zip(
                        local_indices.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()
                    ):
                        predictions[name][local][view_index] = (
                            canonical_label(local_labels, int(predicted_id), canonical), float(score)
                        )
        checkpoint_evidence[name] = {
            "path": str(checkpoint_path.resolve()), "sha256": actual_hash,
            "architecture": architecture, "input_size": input_size,
            "resize_mode": resize_mode, "body_types": local_labels,
        }
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(architecture_families) < 3:
        raise RuntimeError(f"teacher architecture diversity insufficient: {sorted(architecture_families)}")

    frame_strong_label: dict[int, str] = {}
    frame_min_confidence: dict[int, float] = {}
    frame_consensus_label: dict[int, str] = {}
    frame_views: dict[int, str] = {}
    frame_rejections: Counter[str] = Counter()
    track_rows: dict[str, list[int]] = defaultdict(list)
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        track_rows[row["track_group"]].append(local_index)
        teacher_views = [predictions[name][local_index] for name, _ in args.checkpoint]
        flat = [value for values in teacher_views for value in values if value is not None]
        labels = [value[0] for value in flat]
        confidences = [value[1] for value in flat]
        per_teacher_view_consistent = all(
            all(value is not None for value in values)
            and len({value[0] for value in values if value is not None}) == 1
            for values in teacher_views
        )
        all_teachers_consistent = bool(labels) and len(set(labels)) == 1
        label = labels[0] if all_teachers_consistent else "unknown"
        minimum_confidence = min(confidences, default=0.0)
        compatible = coarse_compatible(row.get("source_vehicle_type", ""), label)
        strong = (
            per_teacher_view_consistent and all_teachers_consistent and compatible
            and label != "unknown" and minimum_confidence >= args.confidence
        )
        frame_consensus_label[local_index] = label
        frame_min_confidence[local_index] = minimum_confidence
        frame_views[local_index] = ";".join(
            f"{name}:" + "/".join(value[0] if value else "missing" for value in values)
            for (name, _), values in zip(args.checkpoint, teacher_views)
        )
        if strong:
            frame_strong_label[local_index] = label
        else:
            if not per_teacher_view_consistent:
                frame_rejections["augmentation_conflict"] += 1
            if not all_teachers_consistent:
                frame_rejections["teacher_conflict"] += 1
            if all_teachers_consistent and not compatible:
                frame_rejections["official_coarse_class_conflict"] += 1
            if label == "unknown":
                frame_rejections["unknown_consensus"] += 1
            if minimum_confidence < args.confidence:
                frame_rejections["confidence_below_threshold"] += 1

    qualified_tracks: dict[str, str] = {}
    track_rejections: Counter[str] = Counter()
    for track, local_indices in track_rows.items():
        strong_labels = [frame_strong_label[index] for index in local_indices if index in frame_strong_label]
        label = qualify_track(
            strong_labels, len(local_indices), args.minimum_track_frames, args.minimum_track_strong_fraction
        )
        if label is not None:
            qualified_tracks[track] = label
        else:
            if len(strong_labels) < args.minimum_track_frames:
                track_rejections["insufficient_strong_frames"] += 1
            if strong_labels and len(set(strong_labels)) > 1:
                track_rejections["strong_frame_label_conflict"] += 1
            if len(strong_labels) / len(local_indices) < args.minimum_track_strong_fraction:
                track_rejections["strong_fraction_below_threshold"] += 1

    review_fields = [
        "stage207_teacher_consensus", "stage207_teacher_consensus_class",
        "stage207_teacher_min_confidence", "stage207_teacher_views",
        "stage207_track_consensus", "stage207_track_consensus_class",
        "stage207_track_strong_frames", "stage207_track_total_frames",
        "stage207_training_role",
    ]
    accepted_counts: Counter[str] = Counter()
    accepted_source_counts: Counter[str] = Counter()
    accepted_rows = 0
    for row in rows:
        for field in review_fields:
            row[field] = "validation_isolated_not_opened" if row.get("split") == "validation" else "not_accepted"
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        track = row["track_group"]
        track_label = qualified_tracks.get(track)
        accepted = track_label is not None and frame_strong_label.get(local_index) == track_label
        if accepted:
            accepted_rows += 1
            accepted_counts[track_label] += 1
            accepted_source_counts[row.get("source_vehicle_type", "unknown")] += 1
            row["body_type"] = track_label
            row["body_type_supervised"] = "true"
            row["pseudo_label"] = "true"
            row["pseudo_label_confidence"] = f"{frame_min_confidence[local_index]:.6f}"
            row["review_status"] = "accepted_stage207_five_teacher_four_view_track_consensus"
            row["stage207_teacher_consensus"] = "accepted"
            row["stage207_training_role"] = "research_only_fine_body_pseudolabel"
        else:
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["pseudo_label"] = "false"
            row["pseudo_label_confidence"] = ""
            row["review_status"] = "rejected_to_unknown_unlabeled_night_consistency"
            row["stage207_teacher_consensus"] = "rejected_to_unknown"
            row["stage207_training_role"] = "unlabeled_night_domain_consistency_only"
        row["stage207_teacher_consensus_class"] = frame_consensus_label[local_index]
        row["stage207_teacher_min_confidence"] = f"{frame_min_confidence[local_index]:.6f}"
        row["stage207_teacher_views"] = frame_views[local_index]
        row["stage207_track_consensus"] = "accepted" if track_label is not None else "rejected"
        row["stage207_track_consensus_class"] = track_label or "unknown"
        row["stage207_track_strong_frames"] = str(sum(index in frame_strong_label for index in track_rows[track]))
        row["stage207_track_total_frames"] = str(len(track_rows[track]))

    args.output_root.mkdir(parents=True)
    output_manifest = args.output_root / "stage207-uadetrac-night-multiteacher.csv"
    output_fields = fields + [field for field in review_fields if field not in fields]
    with output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    authorized = accepted_rows >= args.minimum_authorized_rows and len(accepted_counts) >= args.minimum_authorized_classes
    report = {
        "schema_version": "stage207-uadetrac-night-multiteacher-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_pass_pseudolabel_gate" if authorized else "complete_fail_closed_pseudolabel_gate",
        "inputs": {
            "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_path(args.manifest),
            "images_root": str(images_root), "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_path(args.labels),
        },
        "scope": {
            "manifest_rows": len(rows), "train_rows_opened": len(selected),
            "validation_rows_isolated_not_opened": isolated_validation_rows,
            "train_tracks": len(track_rows),
        },
        "results": {
            "frame_strong_rows_before_track_gate": len(frame_strong_label),
            "qualified_tracks": len(qualified_tracks), "accepted_rows": accepted_rows,
            "rejected_to_unknown_rows": len(selected) - accepted_rows,
            "accepted_body_type_counts": dict(sorted(accepted_counts.items())),
            "accepted_source_type_counts": dict(sorted(accepted_source_counts.items())),
            "frame_rejection_reasons_nonexclusive": dict(sorted(frame_rejections.items())),
            "track_rejection_reasons_nonexclusive": dict(sorted(track_rejections.items())),
            "pseudo_label_training_authorized": authorized,
        },
        "policy": {
            "confidence_floor": args.confidence, "views": list(VIEWS),
            "minimum_track_frames": args.minimum_track_frames,
            "minimum_track_strong_fraction": args.minimum_track_strong_fraction,
            "minimum_authorized_rows": args.minimum_authorized_rows,
            "minimum_authorized_classes": args.minimum_authorized_classes,
            "coarse_compatibility": {key: sorted(value) for key, value in COARSE_COMPATIBILITY.items()},
            "all_teachers_exact_agreement": True, "all_views_exact_agreement": True,
            "rejected_rows_remain_unknown": True, "color_labels_all_unknown": True,
            "license_status": "research-only_non-deployable; upstream legal review remains required",
            "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
        "checkpoints": checkpoint_evidence,
        "architecture_families": sorted(architecture_families),
        "outputs": {"manifest": str(output_manifest), "manifest_sha256": sha256_path(output_manifest)},
        "decision": (
            "accepted rows may enter only a research-only body candidate; rejected rows remain unlabeled night consistency"
            if authorized else
            "do not use any Stage207 rows as supervised labels; retain train rows only for unlabeled night consistency"
        ),
    }
    report_path = args.output_root / "stage207-uadetrac-night-multiteacher.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_root / "SHA256SUMS").write_text(
        f"{sha256_path(report_path)}  {report_path.name}\n"
        f"{sha256_path(output_manifest)}  {output_manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
