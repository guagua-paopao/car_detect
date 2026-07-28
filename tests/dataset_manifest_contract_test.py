#!/usr/bin/env python3
"""Contract and negative tests for the M1 dataset manifest validator."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_dataset_manifest import validate_manifest  # noqa: E402


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain an object")
    return value


def require_error(errors: list[str], text: str) -> None:
    if not any(text in error for error in errors):
        raise AssertionError(f"expected error containing {text!r}, got {errors!r}")


def main() -> None:
    labels = load(ROOT / "config" / "vehicle_labels.v1.json")
    manifest = load(ROOT / "data" / "manifests" / "dataset_manifest.v1.example.json")
    schema = load(ROOT / "api" / "schemas" / "dataset_manifest.v1.schema.json")

    errors = validate_manifest(manifest, labels)
    if errors:
        raise AssertionError(f"example manifest must pass: {errors!r}")

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise AssertionError("dataset manifest schema must use JSON Schema 2020-12")
    if schema.get("additionalProperties") is not False:
        raise AssertionError("dataset manifest schema must reject unknown root fields")
    if schema["properties"]["labels_version"]["pattern"] != "^vehicle-labels-v[0-9]+$":
        raise AssertionError("dataset manifest must version the canonical label mapping")

    leaked = copy.deepcopy(manifest)
    leaked["samples"][1]["group"]["track_group"] = leaked["samples"][0]["group"]["track_group"]
    require_error(validate_manifest(leaked, labels), "data leakage")

    absolute_path = copy.deepcopy(manifest)
    absolute_path["samples"][0]["relative_path"] = "C:/private/train.jpg"
    require_error(validate_manifest(absolute_path, labels), "safe POSIX relative path")

    invalid_label = copy.deepcopy(manifest)
    invalid_label["samples"][0]["annotations"]["attributes"]["color"] = "purple"
    require_error(validate_manifest(invalid_label, labels), "color is invalid")

    unapproved = copy.deepcopy(manifest)
    unapproved["sources"][0]["usage_status"] = "review_required"
    errors = validate_manifest(unapproved, labels)
    require_error(errors, "before license/authorization approval")
    require_error(errors, "sources remain unapproved")

    poor_forced = copy.deepcopy(manifest)
    attributes = poor_forced["samples"][0]["annotations"]["attributes"]
    attributes["crop_quality"] = "poor"
    require_error(validate_manifest(poor_forced, labels), "poor crop must use unknown")

    analytics = load(ROOT / "config" / "vehicle_analytics.yaml")
    if analytics["labels_version"] != labels["labels_version"]:
        raise AssertionError("runtime config labels_version must match the canonical labels")
    if analytics["vehicle_analytics"]["body_types"] != labels["body_types"]:
        raise AssertionError("runtime body types drifted from the canonical labels")
    if analytics["vehicle_analytics"]["colors"] != labels["colors"]:
        raise AssertionError("runtime colors drifted from the canonical labels")

    print("PASS: M1 dataset manifest, label mapping, license gate, and leakage tests")


if __name__ == "__main__":
    main()
