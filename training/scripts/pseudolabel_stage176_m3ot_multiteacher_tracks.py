from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint  # noqa: E402


VIEWS = ("original", "horizontal_flip", "dim", "brighten")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use NAME=/absolute/path")
    name, raw = value.split("=", 1)
    path = Path(raw)
    if not name:
        raise argparse.ArgumentTypeError("empty checkpoint name")
    return name, path


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


def qualify_track(strong_labels: list[str], total_rows: int, minimum_frames: int, minimum_fraction: float) -> str | None:
    if total_rows <= 0 or len(strong_labels) < minimum_frames:
        return None
    if len(set(strong_labels)) != 1:
        return None
    if len(strong_labels) / total_rows < minimum_fraction:
        return None
    label = strong_labels[0]
    return label if label != "unknown" else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict five-teacher, four-view, track-consistent M3OT body pseudo-label audit.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_named_path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", action="append", type=parse_named_hash, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--minimum-track-frames", type=int, default=3)
    parser.add_argument("--minimum-track-strong-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-authorized-rows", type=int, default=500)
    parser.add_argument("--minimum-authorized-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) < 4:
        raise ValueError("at least four teachers are required")
    checkpoint_names = [name for name, _ in args.checkpoint]
    expected_hashes = dict(args.expected_checkpoint_sha256)
    if len(set(checkpoint_names)) != len(checkpoint_names) or set(checkpoint_names) != set(expected_hashes):
        raise ValueError("checkpoint names and pinned SHA256 names must match exactly")
    if not 0.0 < args.confidence <= 1.0 or not 0.0 < args.minimum_track_strong_fraction <= 1.0:
        raise ValueError("invalid confidence or track fraction")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage176 evidence")
    if sha256_file(args.manifest).lower() != args.expected_manifest_sha256.lower():
        raise ValueError("input manifest SHA256 mismatch")
    if sha256_file(args.labels).lower() != args.expected_labels_sha256.lower():
        raise ValueError("labels SHA256 mismatch")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    canonical = list(json.loads(args.labels.read_text(encoding="utf-8"))["body_types"])
    if "unknown" not in canonical:
        raise ValueError("canonical body taxonomy has no unknown")
    images_root = args.images_root.resolve()
    if images_root.is_symlink() or not images_root.is_dir():
        raise ValueError("images root missing or symbolic")

    selected: list[int] = []
    resolved_paths: list[Path] = []
    for index, row in enumerate(rows):
        if row.get("split") != "train" or row.get("body_type") != "unknown" or truthy(row.get("body_type_supervised")):
            raise ValueError(f"input row is not unknown-safe train-only: {index + 2}")
        if row.get("color") != "unknown" or truthy(row.get("color_supervised")):
            raise ValueError(f"input color contract violated: {index + 2}")
        basename = Path(row.get("image_path", "")).name
        if not basename or basename != Path(basename).name:
            raise ValueError(f"unsafe image basename: {index + 2}")
        path = (images_root / basename).resolve()
        path.relative_to(images_root)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        selected.append(index)
        resolved_paths.append(path)

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
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash.lower() != expected_hashes[name]:
            raise ValueError(f"checkpoint SHA256 mismatch: {name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        local_labels = list(checkpoint.get("body_types", []))
        if not local_labels or "unknown" not in local_labels or not set(local_labels) <= set(canonical):
            raise ValueError(f"{name} body taxonomy cannot map by exact class name")
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        input_size = int(checkpoint.get("input_size", 224))
        resize_mode = str(checkpoint.get("resize_mode", "stretch"))
        if resize_mode != "stretch":
            raise ValueError(f"{name} uses unsupported review geometry: {resize_mode}")
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
                    for local, predicted_id, score in zip(local_indices.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
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
        raise ValueError(f"teacher architecture diversity insufficient: {sorted(architecture_families)}")

    frame_strong_label: dict[int, str] = {}
    frame_min_confidence: dict[int, float] = {}
    frame_consensus_label: dict[int, str] = {}
    frame_views: dict[int, str] = {}
    rejection_reasons: Counter[str] = Counter()
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
        strong = (
            per_teacher_view_consistent and all_teachers_consistent
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
                rejection_reasons["augmentation_conflict"] += 1
            if not all_teachers_consistent:
                rejection_reasons["teacher_conflict"] += 1
            if label == "unknown":
                rejection_reasons["unknown_consensus"] += 1
            if minimum_confidence < args.confidence:
                rejection_reasons["confidence_below_threshold"] += 1

    qualified_tracks: dict[str, str] = {}
    track_rejections: Counter[str] = Counter()
    for track, local_indices in track_rows.items():
        strong_labels = [frame_strong_label[index] for index in local_indices if index in frame_strong_label]
        label = qualify_track(strong_labels, len(local_indices), args.minimum_track_frames, args.minimum_track_strong_fraction)
        if label is not None:
            qualified_tracks[track] = label
        else:
            if len(strong_labels) < args.minimum_track_frames:
                track_rejections["insufficient_strong_frames"] += 1
            if strong_labels and len(set(strong_labels)) > 1:
                track_rejections["strong_frame_label_conflict"] += 1
            if len(strong_labels) / len(local_indices) < args.minimum_track_strong_fraction:
                track_rejections["strong_fraction_below_threshold"] += 1

    accepted_counts: Counter[str] = Counter()
    accepted_conditions: Counter[str] = Counter()
    accepted_rows = 0
    review_fields = [
        "teacher_consensus_class", "teacher_consensus_confidence", "teacher_consensus_views",
        "teacher_audit_policy", "track_teacher_consensus", "track_teacher_consensus_class",
        "track_teacher_strong_frames", "track_teacher_total_frames",
    ]
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        track = row["track_group"]
        track_label = qualified_tracks.get(track)
        accepted = track_label is not None and frame_strong_label.get(local_index) == track_label
        if accepted:
            accepted_rows += 1
            accepted_counts[track_label] += 1
            accepted_conditions[row.get("lighting", "unknown")] += 1
            row["body_type"] = track_label
            row["body_type_supervised"] = "true"
            row["pseudo_label"] = "true"
            row["pseudo_label_confidence"] = f"{frame_min_confidence[local_index]:.6f}"
            row["teacher_consensus"] = "accepted"
            row["review_status"] = "accepted_stage176_multiteacher_track_consensus"
        else:
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["pseudo_label"] = "false"
            row["pseudo_label_confidence"] = ""
            row["teacher_consensus"] = "rejected_to_unknown"
            row["review_status"] = "rejected_to_unknown_or_unlabeled_consistency"
        row["image_path"] = str(resolved_paths[local_index])
        row["teacher_consensus_class"] = frame_consensus_label[local_index]
        row["teacher_consensus_confidence"] = f"{frame_min_confidence[local_index]:.6f}"
        row["teacher_consensus_views"] = frame_views[local_index]
        row["teacher_audit_policy"] = "all five teachers exact-agree across original/hflip/dim/brighten at confidence floor"
        row["track_teacher_consensus"] = "accepted" if track_label is not None else "rejected"
        row["track_teacher_consensus_class"] = track_label or "unknown"
        row["track_teacher_strong_frames"] = str(sum(index in frame_strong_label for index in track_rows[track]))
        row["track_teacher_total_frames"] = str(len(track_rows[track]))

    output_fields = fields + [field for field in review_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    authorized = accepted_rows >= args.minimum_authorized_rows and len(accepted_counts) >= args.minimum_authorized_classes
    report = {
        "schema_version": "stage176-m3ot-multiteacher-track-consensus-v1",
        "status": "pass_audit_complete",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "images_root": str(images_root),
        "audited_rows": len(selected),
        "track_groups": len(track_rows),
        "frame_strong_rows_before_track_gate": len(frame_strong_label),
        "qualified_tracks": len(qualified_tracks),
        "accepted_rows": accepted_rows,
        "rejected_rows": len(selected) - accepted_rows,
        "accepted_body_type_counts": dict(sorted(accepted_counts.items())),
        "accepted_condition_counts": dict(sorted(accepted_conditions.items())),
        "frame_rejection_reasons_nonexclusive": dict(sorted(rejection_reasons.items())),
        "track_rejection_reasons_nonexclusive": dict(sorted(track_rejections.items())),
        "confidence_threshold": args.confidence,
        "views": list(VIEWS),
        "minimum_track_frames": args.minimum_track_frames,
        "minimum_track_strong_fraction": args.minimum_track_strong_fraction,
        "checkpoints": checkpoint_evidence,
        "architecture_families": sorted(architecture_families),
        "pseudo_label_training_authorized": authorized,
        "authorization_requirements": {
            "minimum_accepted_rows": args.minimum_authorized_rows,
            "minimum_accepted_classes": args.minimum_authorized_classes,
        },
        "policy": {
            "all_teachers_exact_agreement": True,
            "all_views_exact_agreement": True,
            "track_level_consistency_required": True,
            "rejected_rows_remain_unknown": True,
            "color_labels_all_unknown": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "accepted rows may enter a research-only train candidate only when pseudo_label_training_authorized is true; all other rows remain unknown for optional feature consistency",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_manifest, args.output_report):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
