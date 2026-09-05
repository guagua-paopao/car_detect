#!/usr/bin/env python3
"""Export a test-passing Stage107 main-body/truck-specialist/color trio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_final_report(report: dict[str, Any]) -> None:
    if report.get("status") != "pass_backend_eligible":
        raise RuntimeError("Stage107 final test did not authorize backend export")
    gates = report.get("composite_gates")
    if not gates or not all(gates.values()) or report.get("all_composite_gates_pass") is not True:
        raise RuntimeError("Stage107 final test contains a failed composite gate")
    policy = report.get("policy", {})
    required = {
        "test_accessed": True,
        "test_used_for_selection": False,
        "threshold_or_temperature_search": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise RuntimeError(f"Stage107 policy mismatch: {key}")


def validate_checkpoint_contract(
    role: str,
    checkpoint: dict[str, Any],
    labels: dict[str, Any],
) -> None:
    if role == "truck_specialist":
        if checkpoint.get("labels_version") != "vehicle-labels-v2-offline-candidate":
            raise RuntimeError("truck specialist labels version mismatch")
        if checkpoint.get("body_types") != ["light_truck", "heavy_truck", "unknown"]:
            raise RuntimeError("truck specialist body contract mismatch")
        if checkpoint.get("colors") != ["unknown"]:
            raise RuntimeError("truck specialist color contract mismatch")
        return
    if checkpoint.get("labels_version") != labels.get("labels_version"):
        raise RuntimeError(f"{role} labels version mismatch")
    if checkpoint.get("body_types") != labels.get("body_types"):
        raise RuntimeError(f"{role} body contract mismatch")
    if checkpoint.get("colors") != labels.get("colors"):
        raise RuntimeError(f"{role} color contract mismatch")
    if role == "body_main" and checkpoint.get("body_hierarchy") != "truck_family":
        raise RuntimeError("body main checkpoint lost truck-family hierarchy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage107-report", type=Path, required=True)
    parser.add_argument("--expected-stage107-report-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite Stage108 ONNX evidence")
    require_sha(args.stage107_report, args.expected_stage107_report_sha256, "Stage107 report")
    require_sha(args.labels, args.expected_labels_sha256, "label contract")
    final_report = json.loads(args.stage107_report.read_text(encoding="utf-8"))
    validate_final_report(final_report)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    candidate = final_report["candidate"]

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("ONNX export dependencies are unavailable") from error
    from src.multitask_mobilenet_v3 import architecture_from_checkpoint, model_from_checkpoint

    class RuntimeInputModel(nn.Module):
        def __init__(self, base: Any, mean_values: list[float], std_values: list[float]):
            super().__init__()
            self.base = base
            self.register_buffer(
                "mean", torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1)
            )

        def forward(self, images: Any) -> Any:
            return self.base((images - self.mean) / self.std)

    specs = {
        "body_main": (
            Path(candidate["body_checkpoint"]),
            candidate["body_checkpoint_sha256"],
            "body_type",
        ),
        "truck_specialist": (
            Path(candidate["body_specialist_checkpoint"]),
            candidate["body_specialist_checkpoint_sha256"],
            "body_type",
        ),
        "color": (
            Path(candidate["color_checkpoint"]),
            candidate["color_checkpoint_sha256"],
            "color",
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    exports: dict[str, Any] = {}
    input_sizes: set[int] = set()
    for role, (checkpoint_path, checkpoint_sha, consumed_output) in specs.items():
        require_sha(checkpoint_path, checkpoint_sha, f"{role} checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        validate_checkpoint_contract(role, checkpoint, labels)
        input_size = int(checkpoint["input_size"])
        if input_size not in {224, 256}:
            raise RuntimeError(f"unsupported {role} input size: {input_size}")
        input_sizes.add(input_size)
        model = model_from_checkpoint(checkpoint, pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        runtime_model = RuntimeInputModel(
            model,
            list(checkpoint["normalization"]["mean"]),
            list(checkpoint["normalization"]["std"]),
        ).eval()
        output = args.output_dir / f"{role.replace('_', '-')}.onnx"
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
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {"images": np.zeros((2, 3, input_size, input_size), dtype=np.float32)},
        )
        expected_shapes = (
            (2, len(checkpoint["body_types"])),
            (2, len(checkpoint["colors"])),
        )
        actual_shapes = tuple(tuple(array.shape) for array in outputs)
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"unexpected {role} ONNX output shapes: {actual_shapes} != {expected_shapes}"
            )
        exports[role] = {
            "consumed_output": consumed_output,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "architecture": architecture_from_checkpoint(checkpoint),
            "input_size": input_size,
            "body_types": checkpoint["body_types"],
            "colors": checkpoint["colors"],
            "onnx": str(output.resolve()),
            "onnx_sha256": sha256(output),
            "onnx_checker": "pass",
            "onnxruntime_cpu_smoke": "pass",
            "input_contract": "RGB float32 [0,1] with embedded checkpoint normalization",
            "outputs": ["body_type", "color"],
        }
    if len(input_sizes) != 1:
        raise RuntimeError(f"three-branch input-size mismatch: {sorted(input_sizes)}")

    require_sha(args.stage107_report, args.expected_stage107_report_sha256, "Stage107 report")
    require_sha(args.labels, args.expected_labels_sha256, "label contract")
    for role, (checkpoint_path, checkpoint_sha, _) in specs.items():
        require_sha(checkpoint_path, checkpoint_sha, f"{role} checkpoint")
    report = {
        "schema_version": "stage108-three-branch-onnx-export-v1",
        "status": "pass_onnx_exported_candidate_only",
        "stage107_report": str(args.stage107_report.resolve()),
        "stage107_report_sha256": sha256(args.stage107_report),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "input_size": next(iter(input_sizes)),
        "thresholds": {
            "body": candidate["body_threshold"],
            "color": candidate["color_threshold"],
            "truck_subtype": candidate["body_specialist_subtype_threshold"],
        },
        "routing_contract": {
            "body": "body-main.onnx:body_type; route winning truck family through truck-specialist.onnx:body_type",
            "truck_specialist_labels": ["light_truck", "heavy_truck", "unknown"],
            "uncertain_truck_subtype": "unknown",
            "color": "color.onnx:color",
            "unused_outputs_must_not_be_routed": True,
        },
        "exports": exports,
        "policy": {
            "candidate_only": True,
            "test_used_for_export_selection": False,
            "frozen_video_used": False,
            "production_registry_modified": False,
            "production_config_modified": False,
            "deployment_performed": False,
        },
        "backend_status": {
            "onnx_pytorch_parity": "pending",
            "tensorrt_build_and_parity": "pending",
            "three_branch_cpp_contract": "pending",
            "real_engine_smoke": "pending",
        },
    }
    report_path = args.output_dir / "onnx-export-report.json"
    atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
