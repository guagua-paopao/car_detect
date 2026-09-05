#!/usr/bin/env python3
"""Train, validate, export, and fingerprint a YOLO vehicle detector."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import sha256_file, write_json  # noqa: E402


def batch_value(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=batch_value, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--close-mosaic", type=int)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="DET-SMOKE-640")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--dataset-version", default="dataset-pilot-v1")
    parser.add_argument("--code-revision", default=os.environ.get("VCAS_CODE_REVISION", "unknown"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install training/requirements-cloud.txt first") from exc

    started = datetime.now(timezone.utc)
    model = YOLO(str(args.resume or args.model))
    train_kwargs = {
        "project": str(args.project.resolve()),
        "name": args.name,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "amp": True,
        "save_period": args.save_period,
    }
    if args.resume:
        results = model.train(resume=True, **train_kwargs)
    else:
        optional_train_kwargs = {
            "optimizer": args.optimizer,
            "cos_lr": args.cos_lr,
        }
        for key, value in (
            ("lr0", args.lr0),
            ("lrf", args.lrf),
            ("weight_decay", args.weight_decay),
            ("warmup_epochs", args.warmup_epochs),
            ("freeze", args.freeze),
            ("close_mosaic", args.close_mosaic),
        ):
            if value is not None:
                optional_train_kwargs[key] = value
        results = model.train(
            data=str(args.data.resolve()),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            **optional_train_kwargs,
            **train_kwargs,
        )
    save_dir = Path(str(results.save_dir)).resolve()
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(f"training did not produce best.pt/last.pt in {save_dir}")

    best_model = YOLO(str(best_path))
    validation = best_model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        split="test",
        device=args.device,
        plots=True,
        project=str(args.project.resolve()),
        name=f"{args.name}-test",
    )
    exported = Path(
        str(
            best_model.export(
                format="onnx",
                imgsz=args.imgsz,
                batch=1,
                dynamic=False,
                simplify=True,
            )
        )
    ).resolve()
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics reported missing ONNX export: {exported}")

    summary = {
        "schema_version": "1.0",
        "run_id": args.name,
        "run_kind": args.run_kind,
        "kind": "detection",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [sys.executable, *sys.argv],
        "code_revision": args.code_revision,
        "dataset_version": args.dataset_version,
        "model": args.model,
        "data": str(args.data.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "artifacts": {
            "best_pt": str(best_path),
            "best_pt_sha256": sha256_file(best_path),
            "last_pt": str(last_path),
            "last_pt_sha256": sha256_file(last_path),
            "onnx": str(exported),
            "onnx_sha256": sha256_file(exported),
        },
        "metrics": {
            "map50": float(validation.box.map50),
            "map50_95": float(validation.box.map),
            "precision_mean": float(validation.box.mp),
            "recall_mean": float(validation.box.mr),
        },
        "release_eligible": False,
        "release_status": (
            "candidate_pending_conversion_and_release_validation"
            if args.run_kind == "formal"
            else "smoke_only"
        ),
        "note": (
            "Formal training candidate; release still requires threshold calibration, "
            "ONNX/TensorRT regression, model-card review, and release acceptance."
            if args.run_kind == "formal"
            else "Smoke artifact only; do not register as vehicle-det-v1."
        ),
    }
    summary_path = save_dir / (
        "run_summary.json" if args.run_kind == "formal" else "smoke_summary.json"
    )
    write_json(summary_path, summary)
    print(f"PASS: detector {args.run_kind} run complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
