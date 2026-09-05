#!/usr/bin/env python3
"""Hash and seal the two Stage156 full-scale body training candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = ("best.pt", "last.pt", "metrics.json", "model_card.json", "test_metrics.json")
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "stage148", "stage155")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def audit_candidate(path: Path, expected_geometry: tuple[int, str]) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if any(marker in str(path).lower() for marker in FROZEN_MARKERS):
        raise RuntimeError("forbidden test/frozen marker in candidate path")
    artifacts: dict[str, Any] = {}
    for name in REQUIRED_ARTIFACTS:
        artifact = path / name
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        artifacts[name] = {"path": str(artifact), "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}
    card = json.loads((path / "model_card.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    if (card.get("input_size"), card.get("resize_mode")) != expected_geometry:
        raise RuntimeError(f"candidate geometry mismatch: {(card.get('input_size'), card.get('resize_mode'))} != {expected_geometry}")
    if card.get("metrics", {}).get("test", {}).get("status") != "not_run" or test_metrics.get("status") != "not_run":
        raise RuntimeError("training candidate accessed a test split")
    command = " ".join(str(item) for item in card.get("command", []))
    if "--skip-test" not in command or any(marker in command.lower() for marker in FROZEN_MARKERS):
        raise RuntimeError("candidate command isolation failed")
    return {"run_dir": str(path), "geometry": {"input_size": expected_geometry[0], "resize_mode": expected_geometry[1]},
            "validation_metrics": metrics, "artifacts": artifacts, "test_status": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-state", type=Path, required=True)
    parser.add_argument("--expected-manifest-state-sha256", required=True)
    parser.add_argument("--candidate-288-dir", type=Path, required=True)
    parser.add_argument("--candidate-256-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--initialization-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-initialization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage156 training state")
    inputs = {
        "manifest_state": require_hash(args.manifest_state, args.expected_manifest_state_sha256, "manifest state"),
        "trainer": require_hash(args.trainer, args.expected_trainer_sha256, "trainer"),
        "labels": require_hash(args.labels, args.expected_labels_sha256, "labels"),
        "initialization_checkpoint": require_hash(args.initialization_checkpoint, args.expected_initialization_sha256, "initialization checkpoint"),
    }
    manifest_state = json.loads(args.manifest_state.read_text(encoding="utf-8"))
    if manifest_state.get("status") != "pass_training_manifest_ready":
        raise RuntimeError("Stage156 manifest was not training-ready")
    for key in ("test_accessed", "stage148_test_reused", "stage155_holdout_reused", "frozen_video_used", "production_model_modified"):
        if manifest_state.get(key) is not False:
            raise RuntimeError(f"manifest isolation failed: {key}")
    candidates = {
        "convnext_tiny_288_letterbox": audit_candidate(args.candidate_288_dir, (288, "letterbox")),
        "convnext_tiny_256_stretch": audit_candidate(args.candidate_256_dir, (256, "stretch")),
    }
    state = {
        "schema_version": "stage156-fullscale-body-training-state-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "training_complete_pending_stage157_validation",
        "inputs": {"manifest_state": {"path": str(args.manifest_state), "sha256": inputs["manifest_state"]},
                   "trainer": {"path": str(args.trainer), "sha256": inputs["trainer"]},
                   "labels": {"path": str(args.labels), "sha256": inputs["labels"]},
                   "initialization_checkpoint": {"path": str(args.initialization_checkpoint), "sha256": inputs["initialization_checkpoint"]}},
        "candidates": candidates,
        "selection_performed": False,
        "test_accessed": False,
        "stage148_test_reused": False,
        "stage155_holdout_reused": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "candidates": {key: value["validation_metrics"] for key, value in candidates.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
