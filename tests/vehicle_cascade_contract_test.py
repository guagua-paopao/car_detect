#!/usr/bin/env python3
"""Dependency-free source and configuration contracts for M3 cascade runtime."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads(
        (ROOT / "config" / "vehicle_analytics.yaml").read_text(encoding="utf-8")
    )
    if config["config_version"] not in {"vehicle-analytics-m3", "vehicle-analytics-m4"}:
        raise AssertionError("cascade runtime config must be M3-compatible")
    analytics = config["vehicle_analytics"]
    expected = {
        "tracking_iou_threshold": 0.30,
        "min_crop_width_px": 96,
        "min_crop_height_px": 64,
        "min_crop_sharpness": 0.02,
        "min_crop_exposure": 0.08,
        "max_crop_exposure": 0.95,
        "max_crop_occlusion": 0.50,
        "reject_truncated_crops": True,
        "attribute_vote_samples": 3,
        "attribute_max_observations": 5,
    }
    for key, value in expected.items():
        if analytics.get(key) != value:
            raise AssertionError(f"unexpected M3 config {key}: {analytics.get(key)!r}")
    if config["queues"]["attribute_pending_per_track"] != 1:
        raise AssertionError("attribute queue must keep at most one crop per track")
    if config["queues"]["attribute_max_pending"] <= 0:
        raise AssertionError("attribute queue must be globally bounded")

    header = (ROOT / "include" / "business" / "vehicle_cascade_runtime.h").read_text(
        encoding="utf-8"
    )
    implementation = (
        ROOT / "src" / "business" / "vehicle_cascade_runtime.cpp"
    ).read_text(encoding="utf-8")
    model_header = (
        ROOT / "include" / "server" / "vehicle_model_contract.h"
    ).read_text(encoding="utf-8")
    for symbol in (
        "VehicleTracker",
        "VehicleCropQualityGate",
        "VehicleAttributeCandidateQueue",
        "VehicleTrackAttributeAggregator",
        "VehicleCascadeRuntime",
        "run_generation",
        "crop_sequence",
        "AttributeQueueMetrics",
    ):
        if symbol not in header:
            raise AssertionError(f"M3 runtime header is missing {symbol}")
    if "shared_ptr<const OwnedImage>" not in model_header:
        raise AssertionError("async attribute crops must own immutable image storage")
    for guard in (
        "stale or foreign detection result",
        "attribute crop sequence must be strictly increasing",
        "attribute model or labels version changed within a track",
    ):
        if guard not in implementation:
            raise AssertionError(f"M3 runtime is missing guard: {guard}")

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "contract.yml"
    ).read_text(encoding="utf-8")
    if "vehicle_cascade_core" not in cmake or "vehicle_cascade_runtime_test" not in cmake:
        raise AssertionError("M3 runtime must be a CMake library with a test target")
    if "Compile vehicle M3 cascade runtime" not in workflow:
        raise AssertionError("CI must compile and run the M3 C++ contract")

    for relative in (
        "docs/runtime/VEHICLE_CASCADE_RUNTIME.md",
        "docs/development/VCAS_M3_CASCADE_RUNTIME.md",
    ):
        if not (ROOT / relative).is_file():
            raise AssertionError(f"missing M3 traceability document: {relative}")

    print("PASS: M3 cascade runtime configuration, ownership, guards, and CI contracts")


if __name__ == "__main__":
    main()
