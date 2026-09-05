#!/usr/bin/env python3
"""Train a detector candidate without reading test data or exporting release files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def batch_value(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=batch_value, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0003)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-version", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(str(args.model.resolve()))
    results = model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        amp=True,
        save_period=-1,
        project=str(args.project.resolve()),
        name=args.name,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.40,
        degrees=1.5,
        translate=0.10,
        scale=0.50,
        shear=1.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.85,
        mixup=0.05,
    )
    save_dir = Path(str(results.save_dir)).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError(f"missing final checkpoints in {save_dir}")
    payload = {
        "schema_version": "1.0",
        "run_id": args.name,
        "kind": "detection",
        "status": "research_only_validation_pending",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [sys.executable, *sys.argv],
        "dataset_version": args.dataset_version,
        "data": str(args.data.resolve()),
        "initial_model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "seed": args.seed,
        "artifacts": {
            "best_pt": str(best),
            "best_pt_sha256": sha256(best),
            "last_pt": str(last),
            "last_pt_sha256": sha256(last),
        },
        "safety": {
            "test_accessed": False,
            "frozen_video_used": False,
            "onnx_exported": False,
            "deployment_performed": False,
            "production_model_modified": False,
        },
        "environment": {"python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    output = save_dir / "validation_only_run_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (save_dir / "validation_only_run_summary.json.sha256").write_text(
        f"{sha256(output)}  validation_only_run_summary.json\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
