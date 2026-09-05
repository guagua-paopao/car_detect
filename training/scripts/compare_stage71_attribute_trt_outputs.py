#!/usr/bin/env python3
"""Compare TensorRT FP16 outputs with Stage71 ONNX references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def parse_trtexec(path: Path) -> dict[str, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise RuntimeError("trtexec output must be a JSON list")
    outputs: dict[str, np.ndarray] = {}
    for item in document:
        name = str(item.get("name", ""))
        dimensions = [int(value) for value in str(item.get("dimensions", "")).split("x")]
        values = np.asarray(item.get("values", []), dtype=np.float32)
        if not name or not dimensions or values.size != int(np.prod(dimensions)):
            raise RuntimeError(f"malformed trtexec tensor: {name!r}")
        outputs[name] = values.reshape(dimensions)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-manifest", type=Path, required=True)
    parser.add_argument("--expected-parity-manifest-sha256", required=True)
    parser.add_argument("--model", choices=("body", "color"), required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError("refusing to overwrite TensorRT parity evidence")
    require_sha(
        args.parity_manifest,
        args.expected_parity_manifest_sha256,
        "parity manifest",
    )
    if not args.engine.is_file() or args.engine.stat().st_size <= 0:
        raise RuntimeError("TensorRT engine is missing or empty")
    if not args.trt_output.is_file():
        raise RuntimeError("trtexec output is missing")
    manifest = json.loads(args.parity_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass_inputs_prepared":
        raise RuntimeError("parity inputs are not eligible")
    if manifest.get("policy", {}).get("validation_only") is not True:
        raise RuntimeError("parity inputs are not validation-only")
    model = manifest["models"][args.model]
    routed_output = model["routed_output"]
    outputs = parse_trtexec(args.trt_output)
    if set(outputs) != {"body_type", "color"}:
        raise RuntimeError(f"unexpected TensorRT outputs: {sorted(outputs)}")

    metrics: dict[str, Any] = {}
    for output_name in ("body_type", "color"):
        reference_info = model["references"][output_name]
        reference_path = args.parity_manifest.parent / reference_info["path"]
        require_sha(reference_path, reference_info["sha256"], f"{output_name} ONNX reference")
        reference = np.load(reference_path)
        candidate = outputs[output_name]
        if candidate.shape != reference.shape:
            raise RuntimeError(
                f"{output_name} shape mismatch: TRT={candidate.shape}, ONNX={reference.shape}"
            )
        logit_delta = np.abs(candidate - reference)
        reference_probability = softmax(reference.astype(np.float32))
        candidate_probability = softmax(candidate.astype(np.float32))
        probability_delta = np.abs(candidate_probability - reference_probability)
        top1_match_rate = float(
            (candidate.argmax(axis=1) == reference.argmax(axis=1)).mean()
        )
        metrics[output_name] = {
            "shape": list(reference.shape),
            "top1_match_rate": top1_match_rate,
            "maximum_absolute_logit_delta": float(logit_delta.max()),
            "mean_absolute_logit_delta": float(logit_delta.mean()),
            "maximum_absolute_probability_delta": float(probability_delta.max()),
            "mean_absolute_probability_delta": float(probability_delta.mean()),
            "routed": output_name == routed_output,
        }

    routed = metrics[routed_output]
    gates = {
        "routed_top1_match_rate_eq_1": routed["top1_match_rate"] == 1.0,
        "routed_max_probability_delta_lte_0_01": routed[
            "maximum_absolute_probability_delta"
        ]
        <= 0.01,
        "routed_mean_probability_delta_lte_0_002": routed[
            "mean_absolute_probability_delta"
        ]
        <= 0.002,
    }
    passed = all(gates.values())
    require_sha(
        args.parity_manifest,
        args.expected_parity_manifest_sha256,
        "parity manifest",
    )
    report = {
        "schema_version": "stage71-attribute-trt-parity-v1",
        "status": "pass" if passed else "fail_closed",
        "model": args.model,
        "routed_output": routed_output,
        "parity_manifest": str(args.parity_manifest.resolve()),
        "parity_manifest_sha256": sha256(args.parity_manifest),
        "trt_output": str(args.trt_output.resolve()),
        "trt_output_sha256": sha256(args.trt_output),
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256(args.engine),
        "metrics": metrics,
        "gates": gates,
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
    print(json.dumps({"status": report["status"], "gates": gates}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
