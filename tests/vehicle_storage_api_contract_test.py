#!/usr/bin/env python3
"""Source-level M4 vehicle storage/API/Outbox contract gates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require_text(path: str, needles: list[str]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{path} is missing {needle!r}")


def main() -> None:
    require_text(
        "db/postgresql/004_vehicle_events.sql",
        [
            "CREATE TABLE IF NOT EXISTS vision_events",
            "CREATE TABLE IF NOT EXISTS vehicle_track_results",
            "UNIQUE(run_id,track_id)",
            "CREATE TABLE IF NOT EXISTS vehicle_attribute_observations",
            "REFERENCES vision_events(event_id)",
            "VALUES(4,",
        ],
    )
    require_text(
        "include/business/camera_task_repository.h",
        [
            "publishVehicleEvent(",
            "getVehicleEvent(",
            "listVehicleEvents(",
            "insertVehicleAttributeObservation(",
            "deleteExpiredVehicleAttributeObservations(",
            "getVisionEvent(",
        ],
    )
    require_text(
        "src/business/vehicle_event_publisher.cpp",
        [
            "VehicleEventPublisher::publish(",
            "VehicleEventPublisher::persistObservation(",
            "VehicleEventPublisher::eventId(",
            "publishVehicleEvent(",
        ],
    )
    require_text(
        "src/business/camera_task_repository.cpp",
        [
            "INSERT INTO vision_events",
            "INSERT INTO vehicle_track_results",
            "INSERT INTO callback_outbox",
            '"VEHICLE_EVENT_ALREADY_EXISTS"',
            "FROM vehicle_track_results v",
            "JOIN vision_events e",
        ],
    )
    require_text(
        "src/server/callback_delivery_worker.cpp",
        [
            "VisionEventRecord event;",
            "getVisionEvent(",
            "const std::string body = event.payload_json;",
            '"Idempotency-Key", event.event_id',
        ],
    )
    require_text(
        "src/server/camera_task_http_controller.cpp",
        [
            '"/api/v1/cameras/<string>/vehicles/realtime"',
            '"/api/v1/cameras/<string>/vehicle-events"',
            '"/api/v1/vehicle-events/<string>"',
            '"/api/v1/vehicle-events/<string>/snapshot"',
            '"/api/v1/models/status"',
            "resolveCameraArtifact(",
            "vehicleResultJson(",
            "sha256Hex(files[\"engine_sha256\"])",
        ],
    )
    schema = json.loads(
        (ROOT / "api/schemas/vehicle_event.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (ROOT / "api/examples/vehicle_passage.v1.json").read_text(
            encoding="utf-8"
        )
    )
    if schema["properties"]["event_kind"]["const"] != "vehicle_passage":
        raise AssertionError("vehicle schema must stay dedicated to vehicle_passage")
    required = set(schema["required"])
    if not required.issubset(example):
        raise AssertionError("vehicle example is missing required schema fields")
    if example["attributes"]["body_type"]["label"] == "unknown":
        raise AssertionError("vehicle example must demonstrate a stable typed result")
    print("PASS: M4 vehicle storage, API, snapshot, and generic Outbox contracts")


if __name__ == "__main__":
    main()
