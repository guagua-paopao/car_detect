#!/usr/bin/env python3
"""Export the one test-passing Stage71 body/color pair as isolated ONNX files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_final_report(report: dict[str, Any]) -> None:
    if report.get("status") != "pass_backend_eligible":
        raise RuntimeError("final test did not authorize backend export")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("final test report contains a failed gate")
    policy = report.get("policy", {})
    required = {
        "test_accessed": True,
        "test_used_for_selection": False,
        "one_validation_selected_pair_tested": True,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "deployment_paused_by_user": True,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"final-test policy mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test-report", type=Path, required=True)
    parser.add_argument("--expected-final-test-report-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite Stage71 ONNX evidence")
    require_sha(
        args.final_test_report,
        args.expected_final_test_report_sha256,
        "final test report",
    )
    require_sha(args.labels, args.expected_labels_sha256, "label contract")
    final_report = json.loads(args.final_test_report.read_text(encoding="utf-8"))
    validate_final_report(final_report)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("ONNX export dependencies are unavailable") from exc

    from src.multitask_mobilenet_v3 import (
        architecture_from_checkpoint,
        model_from_checkpoint,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    exported: dict[str, Any] = {}
    for head in ("body", "color"):
        checkpoint_path = Path(final_report["candidate"][f"{head}_checkpoint"])
        expected_checkpoint_sha = final_report["candidate"][f"{head}_checkpoint_sha256"]
        require_sha(checkpoint_path, expected_checkpoint_sha, f"{head} checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if list(checkpoint["body_types"]) != list(labels["body_types"]):
            raise RuntimeError(f"{head} checkpoint body labels do not match contract")
        if list(checkpoint["colors"]) != list(labels["colors"]):
            raise RuntimeError(f"{head} checkpoint color labels do not match contract")
        input_size = int(checkpoint["input_size"])
        if input_size not in {224, 256}:
            raise RuntimeError(f"unsupported candidate input size: {input_size}")

        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        mean = checkpoint["normalization"]["mean"]
        std = checkpoint["normalization"]["std"]

        class RuntimeInputModel(nn.Module):
            def __init__(self, base, mean_values, std_values):
                super().__init__()
                self.base = base
                self.register_buffer(
                    "mean",
                    torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1),
                )
                self.register_buffer(
                    "std",
                    torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1),
                )

            def forward(self, images):
                return self.base((images - self.mean) / self.std)

        runtime_model = RuntimeInputModel(model, mean, std).eval()
        output = args.output_dir / f"{head}-specialist.onnx"
        dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
        torch.onnx.export(
            runtime_model,
            dummy,
            output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["body_type", "color"],
            dynamic_axes={
                "images": {0: "batch"},
                "body_type": {0: "batch"},
                "color": {0: "batch"},
            },
        )
        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)
        graph_inputs = [item.name for item in onnx_model.graph.input]
        graph_outputs = [item.name for item in onnx_model.graph.output]
        if graph_inputs != ["images"] or graph_outputs != ["body_type", "color"]:
            raise RuntimeError(
                f"unexpected {head} ONNX contract: {graph_inputs} -> {graph_outputs}"
            )
        session = ort.InferenceSession(
            str(output), providers=["CPUExecutionProvider"]
        )
        body_logits, color_logits = session.run(
            None,
            {
                "images": np.zeros(
                    (2, 3, input_size, input_size), dtype=np.float32
                )
            },
        )
        if body_logits.shape != (2, len(labels["body_types"])):
            raise RuntimeError(f"unexpected {head} body output shape")
        if color_logits.shape != (2, len(labels["colors"])):
            raise RuntimeError(f"unexpected {head} color output shape")
        exported[head] = {
            "specialist_head_consumed": "body_type" if head == "body" else "color",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "architecture": architecture_from_checkpoint(checkpoint),
            "input_size": input_size,
            "onnx": str(output.resolve()),
            "onnx_sha256": sha256(output),
            "onnx_checker": "pass",
            "onnxruntime_cpu_smoke": "pass",
            "input_contract": "RGB float32 [0,1], embedded ImageNet normalization",
            "outputs": ["body_type", "color"],
        }

    # Recheck the authorization and source checkpoints after a potentially
    # long export so evidence cannot be rebound to changed files.
    require_sha(
        args.final_test_report,
        args.expected_final_test_report_sha256,
        "final test report",
    )
    require_sha(args.labels, args.expected_labels_sha256, "label contract")
    for head in ("body", "color"):
        require_sha(
            Path(final_report["candidate"][f"{head}_checkpoint"]),
            final_report["candidate"][f"{head}_checkpoint_sha256"],
            f"{head} checkpoint",
        )

    report = {
        "schema_version": "stage71-decoupled-onnx-export-v1",
        "status": "pass_onnx_exported_candidate_only",
        "final_test_report": str(args.final_test_report.resolve()),
        "final_test_report_sha256": sha256(args.final_test_report),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "selected_pair": final_report["selection"]["pair_id"],
        "thresholds": final_report["candidate"]["thresholds"],
        "head_routing": {
            "body_type": "body-specialist.onnx:body_type",
            "color": "color-specialist.onnx:color",
            "unused_outputs_must_not_be_routed": True,
        },
        "exports": exported,
        "policy": {
            "candidate_only": True,
            "test_used_for_export_selection": False,
            "frozen_video_used": False,
            "production_registry_modified": False,
            "production_config_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
        "backend_status": {
            "onnx_pytorch_parity": "pending",
            "tensorrt_build_and_parity": "pending",
            "cpp_contract": "pending",
            "real_engine_smoke": "pending",
        },
    }
    report_path = args.output_dir / "onnx-export-report.json"
    atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
