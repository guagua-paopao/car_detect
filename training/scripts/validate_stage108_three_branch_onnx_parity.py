#!/usr/bin/env python3
"""Validate PyTorch/ONNX parity for all three Stage108 routed branches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

ROUTES = {
    "body_main": (0, "body_type"),
    "truck_specialist": (0, "body_type"),
    "color": (1, "color"),
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "frozen_video")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_export_report(report: dict[str, Any]) -> None:
    if report.get("status") != "pass_onnx_exported_candidate_only":
        raise RuntimeError("Stage108 export report is not eligible for parity")
    exports = report.get("exports", {})
    if set(exports) != set(ROUTES):
        raise RuntimeError("Stage108 export report does not contain exactly three branches")
    if report.get("policy", {}).get("frozen_video_used") is not False:
        raise RuntimeError("Stage108 export used frozen video")
    for role, (_, consumed) in ROUTES.items():
        if exports[role].get("consumed_output") != consumed:
            raise RuntimeError(f"Stage108 {role} routing output mismatch")


def resolve_validation_rows(
    manifest: Path, safety_root: Path, limit: int
) -> list[dict[str, str]]:
    safety_root = safety_root.resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == "validation"
            and str(row.get("review_status", "approved")).strip().lower() == "approved"
        ][:limit]
    if len(rows) < 512:
        raise RuntimeError(f"only {len(rows)} validation parity rows; at least 512 required")
    for line, row in enumerate(rows, start=2):
        searchable = " ".join(str(value).lower() for value in row.values())
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"frozen marker in parity row {line}")
        raw = Path(row["image_path"])
        resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"parity image escapes datasets root: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["_resolved_image_path"] = str(resolved)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-export-report-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage108 parity evidence")
    if args.limit < 512:
        raise RuntimeError("parity requires at least 512 requested rows")
    require_sha(args.export_report, args.expected_export_report_sha256, "export report")
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    export_report = json.loads(args.export_report.read_text(encoding="utf-8"))
    validate_export_report(export_report)
    rows = resolve_validation_rows(args.manifest, args.datasets_safety_root, args.limit)
    try:
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch/ONNX Runtime dependencies are unavailable") from error
    from src.multitask_mobilenet_v3 import model_from_checkpoint

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    results: dict[str, Any] = {}
    for role, (output_index, consumed_output) in ROUTES.items():
        evidence = export_report["exports"][role]
        checkpoint_path = Path(evidence["checkpoint"])
        onnx_path = Path(evidence["onnx"])
        require_sha(checkpoint_path, evidence["checkpoint_sha256"], f"{role} checkpoint")
        require_sha(onnx_path, evidence["onnx_sha256"], f"{role} ONNX")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if [item.name for item in session.get_outputs()] != ["body_type", "color"]:
            raise RuntimeError(f"unexpected {role} ONNX output order")
        input_name = session.get_inputs()[0].name
        input_size = int(checkpoint["input_size"])
        mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        matches = 0
        samples = 0
        maximum_delta = 0.0
        absolute_sum = 0.0
        element_count = 0
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                images = []
                for row in rows[start : start + args.batch_size]:
                    with Image.open(row["_resolved_image_path"]) as image:
                        resized = image.convert("RGB").resize(
                            (input_size, input_size), Image.Resampling.BILINEAR
                        )
                        array = np.asarray(resized, dtype=np.float32) / 255.0
                    images.append(np.transpose(array, (2, 0, 1)))
                raw = np.ascontiguousarray(images, dtype=np.float32)
                normalized = (raw - mean) / std
                pytorch_logits = model(torch.from_numpy(normalized).to(device))[output_index].cpu().numpy()
                onnx_logits = session.run(None, {input_name: raw})[output_index]
                delta = np.abs(pytorch_logits - onnx_logits)
                maximum_delta = max(maximum_delta, float(delta.max(initial=0.0)))
                absolute_sum += float(delta.sum())
                element_count += int(delta.size)
                matches += int((pytorch_logits.argmax(1) == onnx_logits.argmax(1)).sum())
                samples += len(raw)
        match_rate = matches / samples
        results[role] = {
            "routed_output": consumed_output,
            "samples": samples,
            "top1_matches": matches,
            "top1_match_rate": match_rate,
            "maximum_absolute_logit_delta": maximum_delta,
            "mean_absolute_logit_delta": absolute_sum / element_count,
            "gate_top1_gte_0_995": match_rate >= 0.995,
            "gate_max_abs_logit_delta_lte_0_001": maximum_delta <= 0.001,
        }
    require_sha(args.export_report, args.expected_export_report_sha256, "export report")
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    passed = all(
        item["gate_top1_gte_0_995"] and item["gate_max_abs_logit_delta_lte_0_001"]
        for item in results.values()
    )
    report = {
        "schema_version": "stage108-three-branch-onnx-parity-v1",
        "status": "pass" if passed else "fail_closed",
        "split": "validation",
        "samples": len(rows),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "export_report": str(args.export_report.resolve()),
        "export_report_sha256": sha256(args.export_report),
        "branches": results,
        "gate": passed,
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "branches": results}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
