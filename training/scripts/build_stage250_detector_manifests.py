#!/usr/bin/env python3
"""Build validation-safe, path-only manifests for detector Stage250."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["car", "bus", "truck", "motorcycle", "vehicle", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--bmd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="det-stage250-hard-replay-r1")
    parser.add_argument("--bmd-train-stride", type=int, default=2)
    return parser.parse_args()


def images(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if not indexes:
        raise ValueError(f"image path has no images component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def hard_base_images(base_images: list[Path]) -> tuple[list[Path], Counter[str]]:
    selected: list[Path] = []
    reasons: Counter[str] = Counter()
    for image in base_images:
        label = label_for(image)
        if not label.is_file():
            raise FileNotFoundError(label)
        rare = False
        small = False
        for line in label.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"invalid YOLO row in {label}: {line}")
            class_id = int(parts[0])
            width, height = float(parts[3]), float(parts[4])
            rare = rare or class_id in {1, 2, 3, 5}
            small = small or width * height < 0.01
        if rare or small:
            selected.append(image)
            if rare:
                reasons["rare_class"] += 1
            if small:
                reasons["small_box"] += 1
    return selected, reasons


def bmd_sequence_and_frame(path: Path) -> tuple[str, int]:
    match = re.match(r"(.+)_([0-9]+)$", path.stem)
    if not match:
        return path.stem, 0
    return match.group(1), int(match.group(2))


def temporal_thin(paths: list[Path], stride: int) -> list[Path]:
    if stride < 1:
        raise ValueError("stride must be positive")
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in paths:
        sequence, frame = bmd_sequence_and_frame(path)
        groups[sequence].append((frame, path))
    selected: list[Path] = []
    for sequence in sorted(groups):
        ordered = sorted(groups[sequence])
        selected.extend(path for index, (_, path) in enumerate(ordered) if index % stride == 0)
    return sorted(selected)


def deterministic_sample(paths: list[Path], count: int, seed: str) -> list[Path]:
    ranked = sorted(
        paths,
        key=lambda path: hashlib.sha256(f"{seed}\0{path.name}".encode()).digest(),
    )
    return sorted(ranked[:count])


def write_lines(path: Path, values: list[Path]) -> str:
    payload = "".join(f"{value.as_posix()}\n" for value in values)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def write_yaml(path: Path, train: Path | None, val: Path) -> None:
    # Ultralytics requires both keys even for val-only calls. Evaluation YAMLs
    # therefore point the unused train key at the same validation manifest.
    train_path = train or val
    rows = ["path: /", f"train: {train_path.as_posix()}", f"val: {val.as_posix()}", "names:"]
    rows.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    base_root = args.base_root.resolve()
    bmd_root = args.bmd_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_train = images(base_root / "images" / "train")
    base_val = images(base_root / "images" / "validation")
    bmd_train_all = images(bmd_root / "images" / "bmd_train")
    bmd_val_all = images(bmd_root / "images" / "bmd_validation")
    base_hard, hard_reasons = hard_base_images(base_train)
    bmd_train = temporal_thin(bmd_train_all, args.bmd_train_stride)
    bmd_val_balanced = deterministic_sample(bmd_val_all, len(base_val), args.seed)

    for collection in (base_train, base_val, base_hard, bmd_train, bmd_val_all):
        for image in collection:
            if not label_for(image).is_file():
                raise FileNotFoundError(label_for(image))

    train_rows = [*base_train, *base_hard, *bmd_train]
    balanced_val_rows = [*base_val, *bmd_val_balanced]
    train_txt = output_root / "train-hard-replay.txt"
    balanced_val_txt = output_root / "validation-balanced.txt"
    base_val_txt = output_root / "validation-base.txt"
    bmd_val_txt = output_root / "validation-bmd45.txt"
    hashes = {
        "train": write_lines(train_txt, train_rows),
        "balanced_validation": write_lines(balanced_val_txt, balanced_val_rows),
        "base_validation": write_lines(base_val_txt, base_val),
        "bmd45_validation": write_lines(bmd_val_txt, bmd_val_all),
    }
    write_yaml(output_root / "train-balanced.yaml", train_txt, balanced_val_txt)
    write_yaml(output_root / "base-validation.yaml", None, base_val_txt)
    write_yaml(output_root / "bmd45-validation.yaml", None, bmd_val_txt)

    report = {
        "schema_version": "1.0",
        "stage": "DET-STAGE250-HARD-REPLAY-R1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": {
            "path_only_no_image_copy": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "bmd_temporal_stride": args.bmd_train_stride,
            "base_hard_replay_rule": "rare class {bus,truck,motorcycle,other} or normalized box area < 0.01",
            "balanced_selection_validation": "all base validation plus deterministic BMD45 validation sample of equal image count",
        },
        "counts": {
            "base_train_unique": len(base_train),
            "base_hard_replay_rows": len(base_hard),
            "bmd45_train_before_thinning": len(bmd_train_all),
            "bmd45_train_after_thinning": len(bmd_train),
            "train_rows": len(train_rows),
            "train_unique_images": len(set(train_rows)),
            "base_validation": len(base_val),
            "bmd45_validation_full": len(bmd_val_all),
            "bmd45_validation_selection_sample": len(bmd_val_balanced),
            "balanced_selection_validation_rows": len(balanced_val_rows),
        },
        "hard_replay_reasons": dict(hard_reasons),
        "manifest_sha256": hashes,
    }
    report_path = output_root / "manifest-audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output_root / "manifest-audit.json.sha256").write_text(
        hashlib.sha256(report_path.read_bytes()).hexdigest() + "  manifest-audit.json\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
