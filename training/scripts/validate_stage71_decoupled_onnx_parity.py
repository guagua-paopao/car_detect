#!/usr/bin/env python3
"""Validate Stage71 PyTorch/ONNX parity for routed body and color heads."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-export-report-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError("refusing to overwrite ONNX parity evidence")
    if args.limit < 512:
        raise RuntimeError("parity requires at least 512 requested samples")
    require_sha(
        args.export_report,
        args.expected_export_report_sha256,
        "ONNX export report",
    )
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    export_report = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export_report.get("status") != "pass_onnx_exported_candidate_only":
        raise RuntimeError("ONNX export report is not eligible for parity")
    if export_report.get("policy", {}).get("frozen_video_used") is not False:
        raise RuntimeError("ONNX export policy violation")

    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch/ONNX Runtime dependencies are unavailable") from exc
    from src.multitask_mobilenet_v3 import model_from_checkpoint

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == args.split
            and (row.get("review_status", "approved") or "approved") == "approved"
        ][: args.limit]
    if len(rows) < 512:
        raise RuntimeError(f"only {len(rows)} parity rows; at least 512 are required")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    results: dict[str, Any] = {}
    for head, output_index in (("body", 0), ("color", 1)):
        evidence = export_report["exports"][head]
        checkpoint_path = Path(evidence["checkpoint"])
        onnx_path = Path(evidence["onnx"])
        require_sha(checkpoint_path, evidence["checkpoint_sha256"], f"{head} checkpoint")
        require_sha(onnx_path, evidence["onnx_sha256"], f"{head} ONNX")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = model_from_checkpoint(checkpoint, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if [item.name for item in session.get_outputs()] != ["body_type", "color"]:
            raise RuntimeError(f"unexpected {head} ONNX output order")
        input_name = session.get_inputs()[0].name
        input_size = int(checkpoint["input_size"])
        mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        matches = 0
        samples = 0
        maximum_absolute_delta = 0.0
        mean_absolute_delta_sum = 0.0
        element_count = 0
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                raw_images = []
                for row in rows[start : start + args.batch_size]:
                    with Image.open(args.manifest.parent / row["image_path"]) as image:
                        resized = image.convert("RGB").resize(
                            (input_size, input_size),
                            Image.Resampling.BILINEAR,
                        )
                        array = np.asarray(resized, dtype=np.float32) / 255.0
                    raw_images.append(np.transpose(array, (2, 0, 1)))
                raw = np.ascontiguousarray(raw_images, dtype=np.float32)
                normalized = (raw - mean) / std
                pytorch_outputs = model(torch.from_numpy(normalized).to(device))
                pytorch_logits = pytorch_outputs[output_index].cpu().numpy()
                onnx_logits = session.run(None, {input_name: raw})[output_index]
                delta = np.abs(pytorch_logits - onnx_logits)
                maximum_absolute_delta = max(
                    maximum_absolute_delta,
                    float(delta.max(initial=0.0)),
                )
                mean_absolute_delta_sum += float(delta.sum())
                element_count += int(delta.size)
                pytorch_top1 = pytorch_logits.argmax(axis=1)
                onnx_top1 = np.asarray(onnx_logits).argmax(axis=1)
                matches += int((pytorch_top1 == onnx_top1).sum())
                samples += len(raw)
        match_rate = matches / samples
        results[head] = {
            "routed_output": "body_type" if head == "body" else "color",
            "samples": samples,
            "top1_matches": matches,
            "top1_match_rate": match_rate,
            "maximum_absolute_logit_delta": maximum_absolute_delta,
            "mean_absolute_logit_delta": (
                mean_absolute_delta_sum / element_count if element_count else None
            ),
            "gate_top1_gte_0_995": match_rate >= 0.995,
            "gate_max_abs_logit_delta_lte_0_001": maximum_absolute_delta <= 0.001,
        }

    require_sha(
        args.export_report,
        args.expected_export_report_sha256,
        "ONNX export report",
    )
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    passed = all(
        item["gate_top1_gte_0_995"]
        and item["gate_max_abs_logit_delta_lte_0_001"]
        for item in results.values()
    )
    report = {
        "schema_version": "stage71-decoupled-onnx-parity-v1",
        "status": "pass" if passed else "fail_closed",
        "split": args.split,
        "samples": len(rows),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "export_report": str(args.export_report.resolve()),
        "export_report_sha256": sha256(args.export_report),
        "heads": results,
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
    print(json.dumps({"status": report["status"], "heads": results}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
