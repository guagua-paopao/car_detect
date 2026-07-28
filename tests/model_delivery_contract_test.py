#!/usr/bin/env python3
"""Tests for model delivery blocking and precision regression gates."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_model_delivery import audit  # noqa: E402
from compare_model_metrics import compare  # noqa: E402


def main() -> None:
    registry = json.loads(
        (ROOT / "models" / "manifests" / "model_registry.v1.json").read_text(
            encoding="utf-8"
        )
    )
    issues = audit(registry, ROOT)
    if not any("missing onnx file" in issue for issue in issues):
        raise AssertionError("planned registry must report missing ONNX payloads")
    if not any("missing engine file" in issue for issue in issues):
        raise AssertionError("planned registry must report missing Engine payloads")
    if not any("delivery_status='planned'" in issue for issue in issues):
        raise AssertionError("planned artifact must not pass the delivery gate")

    reference = {
        "schema_version": "1.0",
        "artifact_id": "vehicle-attr-v1",
        "backend": "native",
        "dataset_version": "dataset-test-v1",
        "labels_version": "vehicle-labels-v1",
        "sample_count": 1000,
        "metrics": {"body_macro_f1": 0.90, "color_macro_f1": 0.87},
        "predictions_sha256": "a" * 64,
    }
    passing = copy.deepcopy(reference)
    passing["backend"] = "tensorrt"
    passing["metrics"]["body_macro_f1"] = 0.896
    passing["metrics"]["color_macro_f1"] = 0.866
    passing["predictions_sha256"] = "b" * 64
    if compare(reference, passing, 0.005):
        raise AssertionError("0.4 percentage-point drop must pass")

    failing = copy.deepcopy(passing)
    failing["metrics"]["color_macro_f1"] = 0.86
    errors = compare(reference, failing, 0.005)
    if not any("color_macro_f1 dropped" in error for error in errors):
        raise AssertionError("1 percentage-point drop must fail")

    print("PASS: model asset blocking and 0.5-point precision regression gates")


if __name__ == "__main__":
    main()
