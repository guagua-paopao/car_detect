#!/usr/bin/env python3
"""Dependency-free checks for the M0 vehicle configuration and event contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "api" / "schemas" / "vehicle_event.v1.schema.json"
EXAMPLE_PATH = ROOT / "api" / "examples" / "vehicle_passage.v1.json"
CONFIG_PATH = ROOT / "config" / "vehicle_analytics.yaml"
LABELS_PATH = ROOT / "config" / "vehicle_labels.v1.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return value


def require_confidence(value: object, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AssertionError(f"{field} must be numeric")
    if not 0 <= float(value) <= 1:
        raise AssertionError(f"{field} must be between 0 and 1")


def require_classification(
    value: object,
    *,
    field: str,
    labels: list[str],
    with_stability: bool,
) -> None:
    if not isinstance(value, dict):
        raise AssertionError(f"{field} must be an object")
    expected = {"label", "confidence"}
    if with_stability:
        expected |= {"stable", "samples_used"}
    if set(value) != expected:
        raise AssertionError(f"{field} fields must be {sorted(expected)}")
    if value["label"] not in labels:
        raise AssertionError(f"{field}.label is not in the frozen label set")
    require_confidence(value["confidence"], f"{field}.confidence")
    if with_stability:
        if not isinstance(value["stable"], bool):
            raise AssertionError(f"{field}.stable must be boolean")
        if not isinstance(value["samples_used"], int) or not 0 <= value["samples_used"] <= 64:
            raise AssertionError(f"{field}.samples_used must be an integer in [0, 64]")


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    example = load_json(EXAMPLE_PATH)
    # JSON is a valid YAML 1.2 subset. Keeping the M0 config JSON-compatible
    # gives CI a dependency-free parser while yaml-cpp can consume the file.
    config = load_json(CONFIG_PATH)
    labels = load_json(LABELS_PATH)

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise AssertionError("vehicle event schema must use JSON Schema 2020-12")
    if schema.get("additionalProperties") is not False:
        raise AssertionError("vehicle event root must reject unknown properties")

    required = set(schema["required"])
    if not required.issubset(example):
        raise AssertionError(f"example is missing required fields: {sorted(required - set(example))}")
    if set(example) - set(schema["properties"]):
        raise AssertionError("example contains properties not declared by the schema")
    if example["schema_version"] != "1.0" or example["event_kind"] != "vehicle_passage":
        raise AssertionError("example must use vehicle_passage schema v1.0")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", example["event_id"]):
        raise AssertionError("example event_id violates the public identifier contract")

    body_labels = config["vehicle_analytics"]["body_types"]
    color_labels = config["vehicle_analytics"]["colors"]
    if config["labels_version"] != labels["labels_version"]:
        raise AssertionError("runtime config labels_version must match the canonical labels")
    if body_labels != labels["body_types"] or color_labels != labels["colors"]:
        raise AssertionError("runtime label lists must match the canonical labels")
    versions = example["model_versions"]
    if versions["config"] != config["config_version"]:
        raise AssertionError("event example config version must match runtime config")
    if versions["labels"] != labels["labels_version"]:
        raise AssertionError("event example labels version must match canonical labels")
    if versions["detector"] != config["models"]["vehicle_detector"]["artifact"]:
        raise AssertionError("event detector version must match runtime model selection")
    if versions["attribute"] != config["models"]["vehicle_attribute"]["artifact"]:
        raise AssertionError("event attribute version must match runtime model selection")
    schema_body_labels = schema["$defs"]["body_type_classification"]["properties"]["label"]["enum"]
    schema_color_labels = schema["$defs"]["color_classification"]["properties"]["label"]["enum"]
    if body_labels != schema_body_labels:
        raise AssertionError("body type labels must match between config and event schema")
    if color_labels != schema_color_labels:
        raise AssertionError("color labels must match between config and event schema")

    vehicle_labels = schema["$defs"]["vehicle_classification"]["properties"]["label"]["enum"]
    require_classification(
        example["vehicle_class"],
        field="vehicle_class",
        labels=vehicle_labels,
        with_stability=False,
    )
    require_classification(
        example["attributes"]["body_type"],
        field="attributes.body_type",
        labels=body_labels,
        with_stability=True,
    )
    require_classification(
        example["attributes"]["color"],
        field="attributes.color",
        labels=color_labels,
        with_stability=True,
    )

    analytics = config["vehicle_analytics"]
    if not 0 < analytics["detection_fps"] <= 120:
        raise AssertionError("detection_fps must be in (0, 120]")
    if not 0 < analytics["detection_confidence_threshold"] <= 1:
        raise AssertionError("detection_confidence_threshold must be in (0, 1]")
    if analytics["min_confirm_hits"] < 1:
        raise AssertionError("min_confirm_hits must be positive")
    if not 0 < analytics["type_threshold"] <= 1:
        raise AssertionError("type_threshold must be in (0, 1]")
    if not 0 < analytics["color_threshold"] <= 1:
        raise AssertionError("color_threshold must be in (0, 1]")
    if config["queues"]["detection_pending_per_camera"] != 1:
        raise AssertionError("M0 freezes latest-frame semantics at one pending frame per camera")
    if config["queues"]["attribute_pending_per_track"] != 1:
        raise AssertionError("M0 freezes attribute deduplication at one pending crop per track")

    serialized = json.dumps(config, ensure_ascii=False).lower()
    forbidden = ("rtsp://", "password", "secret", "callback_url", "postgres_dsn")
    for token in forbidden:
        if token in serialized:
            raise AssertionError(f"vehicle config must not contain secret-bearing field: {token}")

    for relative in (
        "docs/DECISIONS.md",
        "docs/TRACEABILITY.md",
        "docs/development/VCAS_M0_BASELINE.md",
        "docs/development/VCAS_M1_DATA_SPEC.md",
        "docs/development/VCAS_M2_MODEL_INTERFACES.md",
        "docs/development/VCAS_M3_CASCADE_RUNTIME.md",
    ):
        if not (ROOT / relative).is_file():
            raise AssertionError(f"missing traceability document: {relative}")

    traceability = (ROOT / "docs" / "TRACEABILITY.md").read_text(encoding="utf-8")
    for requirement_id in ("VCAS-FR-001", "VCAS-FR-007", "VCAS-NFR-004"):
        if requirement_id not in traceability:
            raise AssertionError(f"traceability matrix is missing {requirement_id}")

    print("PASS: vehicle M0 config, labels, event schema, example, and traceability contracts")


if __name__ == "__main__":
    main()
