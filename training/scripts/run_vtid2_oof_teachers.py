#!/usr/bin/env python3
"""Sequentially train nested-holdout VTID2 teachers with auditable state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-prefix", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite OOF teacher root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / "run-state.json"
    train_script = project_root / "training" / "scripts" / "train_attribute.py"
    labels = project_root / "config" / "vehicle_labels.v1.json"
    runs = []
    state = {
        "schema_version": "vtid2-oof-teacher-runs-v1",
        "created_at": now(),
        "status": "running",
        "folds": args.folds,
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "init_checkpoint_sha256": sha256(args.init_checkpoint),
        "train_script": str(train_script),
        "train_script_sha256": sha256(train_script),
        "frozen_video_used": False,
        "test_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "runs": runs,
    }
    atomic_json(state_path, state)
    try:
        for fold in range(args.folds):
            manifest = Path(f"{args.manifest_prefix}.fold{fold}.csv").resolve()
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            output = args.output_root / f"fold{fold}"
            command = [
                sys.executable, str(train_script),
                "--manifest", str(manifest),
                "--labels", str(labels),
                "--input-size", "256",
                "--architecture", "mobilenet_v3_large",
                "--resize-mode", "stretch",
                "--epochs", str(args.epochs),
                "--batch-size", "96",
                "--workers", "4",
                "--learning-rate", "0.0001",
                "--weight-decay", "0.0001",
                "--body-loss-weight", "1.0",
                "--color-loss-weight", "0.0",
                "--focal-gamma", "1.5",
                "--color-focal-gamma", "0.0",
                "--class-weighting", "inverse_sqrt",
                "--label-smoothing", "0.03",
                "--freeze-backbone-epochs", "1",
                "--patience", "3",
                "--augmentation-profile", "hard_scene",
                "--selection-head", "body",
                "--hard-sample-weight", "0.2",
                "--small-sample-weight", "0.2",
                "--run-kind", "formal",
                "--dataset-version", "vtid2-oof-audit-v1",
                "--code-revision", "vtid2-oof-audit-v1",
                "--device", args.device,
                "--init-checkpoint", str(args.init_checkpoint.resolve()),
                "--output-dir", str(output),
                "--skip-test",
            ]
            run = {
                "fold": fold,
                "status": "running",
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "output": str(output),
                "command": command,
                "started_at": now(),
            }
            runs.append(run)
            atomic_json(state_path, state)
            log_path = args.output_root / f"fold{fold}.launcher.log"
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                result = subprocess.run(command, cwd=project_root, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                run.update(status="failed", returncode=result.returncode, finished_at=now(), log=str(log_path))
                state.update(status="failed", updated_at=now())
                atomic_json(state_path, state)
                return result.returncode
            best = output / "best.pt"
            metrics = output / "metrics.json"
            run.update(
                status="complete", returncode=0, finished_at=now(), log=str(log_path),
                best_checkpoint=str(best), best_checkpoint_sha256=sha256(best),
                metrics=str(metrics), metrics_sha256=sha256(metrics),
            )
            atomic_json(state_path, state)
        state.update(status="complete_validation_only", updated_at=now())
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        state.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
