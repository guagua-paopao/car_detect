#!/usr/bin/env python3
"""Source-level guards for the real TensorRT vehicle adapters."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    header = (
        ROOT / "include" / "server" / "vehicle_tensorrt_adapters.h"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "server" / "vehicle_tensorrt_adapters.cpp"
    ).read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    for symbol in (
        "TensorRtVehicleDetectionRunner",
        "TensorRtVehicleAttributeRunner",
        "TensorRtDetectionOptions",
        "TensorRtAttributeOptions",
    ):
        if symbol not in header:
            raise AssertionError(f"TensorRT adapter header is missing {symbol}")
    for operation in (
        "deserializeCudaEngine",
        "setInputShape",
        "setInputTensorAddress",
        "setOutputTensorAddress",
        "enqueueV3",
        "cudaMemcpyAsync",
        "EVP_sha256",
        "decodeYolo",
        "softmaxTop1",
    ):
        if operation not in source:
            raise AssertionError(f"TensorRT adapter is missing real operation {operation}")
    for guard in (
        "engine SHA256 does not match model registry",
        "must be engine_validated or deployed",
        "attribute output volumes do not match batch and labels",
    ):
        if guard not in source:
            raise AssertionError(f"TensorRT adapter is missing gate: {guard}")
    forbidden = ("TODO", "fake inference", "placeholder output")
    for token in forbidden:
        if token.lower() in source.lower():
            raise AssertionError(f"TensorRT adapter contains forbidden placeholder: {token}")
    if "vehicle_tensorrt_adapters" not in cmake:
        raise AssertionError("CMake must build the real TensorRT adapter library")
    if "vehicle_tensorrt_adapter_contract_test" not in cmake:
        raise AssertionError("CMake must build the TensorRT adapter contract test")

    print("PASS: real TensorRT adapter source, CUDA execution, SHA256, and decode gates")


if __name__ == "__main__":
    main()
