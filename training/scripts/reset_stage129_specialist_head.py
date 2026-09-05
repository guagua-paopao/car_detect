#!/usr/bin/env python3
"""Preserve a specialist backbone while deterministically resetting its body head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import model_from_checkpoint  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(state: dict, excluded_prefix: str = "") -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if excluded_prefix and key.startswith(excluded_prefix):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reset_body_head(checkpoint: dict, seed: int) -> tuple[dict, dict]:
    import torch

    if checkpoint.get("body_types") != ["light_truck", "heavy_truck", "unknown"]:
        raise RuntimeError("specialist body taxonomy mismatch")
    if checkpoint.get("colors") != ["unknown"]:
        raise RuntimeError("specialist color taxonomy mismatch")
    model = model_from_checkpoint(checkpoint, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.manual_seed(seed)
    model.body_type_head.reset_parameters()
    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    changed = [key for key in sorted(before) if not torch.equal(before[key], after[key])]
    unchanged_non_head = all(
        torch.equal(before[key], after[key])
        for key in before if not key.startswith("body_type_head.")
    )
    if not changed or not all(key.startswith("body_type_head.") for key in changed):
        raise RuntimeError(f"unexpected reset keys: {changed}")
    if not unchanged_non_head:
        raise RuntimeError("non-head parameter changed")
    output = {
        key: value for key, value in checkpoint.items()
        if key not in {"optimizer_state", "scheduler_state", "epoch", "best_score", "gate_best_key", "epochs_without_improvement"}
    }
    output["model_state"] = after
    output["stage129_init_policy"] = "preserve_stage99_backbone_reset_body_type_head"
    output["stage129_reset_seed"] = seed
    report = {
        "changed_keys": changed,
        "unchanged_non_head": unchanged_non_head,
        "backbone_digest_before": tensor_digest(before, "body_type_head."),
        "backbone_digest_after": tensor_digest(after, "body_type_head."),
        "body_head_digest_before": tensor_digest({k: v for k, v in before.items() if k.startswith("body_type_head.")}),
        "body_head_digest_after": tensor_digest({k: v for k, v in after.items() if k.startswith("body_type_head.")}),
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage129 initialization evidence")
    actual = sha256(args.input_checkpoint)
    if actual.lower() != args.expected_input_sha256.lower():
        raise RuntimeError(f"input SHA256 mismatch: expected={args.expected_input_sha256} actual={actual}")
    import torch
    checkpoint = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
    output, report = reset_body_head(checkpoint, args.seed)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(output, args.output_checkpoint)
    full_report = {
        "schema_version": "stage129-reset-specialist-head-v1",
        "status": "pass",
        "input_checkpoint": str(args.input_checkpoint.resolve()),
        "input_checkpoint_sha256": actual,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256(args.output_checkpoint),
        "seed": args.seed,
        **report,
        "policy": {
            "stage99_visual_backbone_preserved": True,
            "stage99_body_head_discarded": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    if full_report["backbone_digest_before"] != full_report["backbone_digest_after"]:
        raise RuntimeError("backbone digest changed")
    if full_report["body_head_digest_before"] == full_report["body_head_digest_after"]:
        raise RuntimeError("body head digest did not change")
    args.output_report.write_text(json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_checkpoint, args.output_report):
        Path(str(path) + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "changed_keys": report["changed_keys"], "output_sha256": full_report["output_checkpoint_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
