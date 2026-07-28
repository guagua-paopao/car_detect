#!/usr/bin/env python3
"""M5 release policy, evidence, packaging, and fail-closed contract tests."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_vehicle_release_evidence import validate  # noqa: E402


def require_text(path: str, needles: list[str]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{path} is missing {needle!r}")


def passing_evidence(pending: dict) -> dict:
    value = copy.deepcopy(pending)
    value["release_id"] = "vehicle_v1_0_0"
    value["status"] = "passed"
    value["source"] = {"commit": "a" * 40, "dirty": False}
    value["versions"]["dataset"] = "dataset-v1"
    value["versions"]["dataset_manifest"] = (
        "data/manifests/dataset_manifest.v1.json"
    )
    value["artifact_hashes"] = [
        {"path": "config/server.yaml", "sha256": "b" * 64},
        {"path": "engines/vehicle-det-v1.engine", "sha256": "c" * 64},
        {
            "path": "data/manifests/dataset_manifest.v1.json",
            "sha256": "e" * 64,
        },
    ]
    value["model_delivery"] = {
        "status": "passed",
        "evidence_path": "reports/models/audit.json",
    }
    value["performance"] = {
        "status": "passed",
        "duration_minutes": 30,
        "queue_growth_detected": False,
        "cameras": [
            {
                "camera_id": "gate_01",
                "width": 1920,
                "height": 1080,
                "detection_fps": 8.1,
            },
            {
                "camera_id": "gate_02",
                "width": 1920,
                "height": 1080,
                "detection_fps": 8.0,
            },
        ],
        "evidence_path": "reports/performance/summary.json",
    }
    value["latency"] = {
        "status": "passed",
        "attribute_stable_p95_ms": 1500,
        "event_persist_p95_ms": 2000,
        "evidence_path": "reports/latency/summary.json",
    }
    value["stability"] = {
        "status": "passed",
        "duration_hours": 8,
        "crashes": 0,
        "gpu_memory_growth_detected": False,
        "unbounded_reconnect_detected": False,
        "stale_run_results_detected": False,
        "evidence_path": "reports/soak/summary.json",
    }
    value["idempotency"] = {
        "status": "passed",
        "duplicate_final_events": 0,
        "evidence_path": "reports/idempotency/summary.json",
    }
    value["fault_drills"] = {
        name: {
            "status": "passed",
            "evidence_path": f"reports/faults/{name}.json",
        }
        for name in (
            "rtsp_disconnect",
            "postgresql_unavailable",
            "redis_restart",
            "callback_5xx",
            "worker_restart",
        )
    }
    value["rollback"] = {
        "database_backup_restore": "passed",
        "previous_binary_retained": "passed",
        "previous_config_retained": "passed",
        "previous_engine_retained": "passed",
        "rollback_drill": "passed",
        "evidence_path": "reports/rollback/summary.json",
    }
    referenced_paths = {
        value["model_delivery"]["evidence_path"],
        value["performance"]["evidence_path"],
        value["latency"]["evidence_path"],
        value["stability"]["evidence_path"],
        value["idempotency"]["evidence_path"],
        value["rollback"]["evidence_path"],
        *(
            gate["evidence_path"]
            for gate in value["fault_drills"].values()
        ),
    }
    value["artifact_hashes"].extend(
        {"path": path, "sha256": "d" * 64}
        for path in sorted(referenced_paths)
    )
    return value


def main() -> None:
    policy = json.loads(
        (ROOT / "config" / "vehicle_release_acceptance.v1.json").read_text(
            encoding="utf-8"
        )
    )
    pending = json.loads(
        (
            ROOT
            / "api"
            / "examples"
            / "vehicle_release_evidence.pending.v1.json"
        ).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            ROOT
            / "api"
            / "schemas"
            / "vehicle_release_evidence.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    if schema["properties"]["status"]["enum"] != [
        "pending",
        "passed",
        "failed",
    ]:
        raise AssertionError("release evidence status contract drifted")
    pending_issues = validate(policy, pending)
    if not any("release status" in issue for issue in pending_issues):
        raise AssertionError("pending evidence must fail the release gate")
    if not any("model_delivery" in issue for issue in pending_issues):
        raise AssertionError("pending models must block release")

    complete = passing_evidence(pending)
    issues = validate(policy, complete)
    if issues:
        raise AssertionError(f"complete boundary evidence must pass: {issues}")

    below_fps = copy.deepcopy(complete)
    below_fps["performance"]["cameras"][1]["detection_fps"] = 7.99
    if not any("detection_fps" in issue for issue in validate(policy, below_fps)):
        raise AssertionError("per-camera FPS below 8 must block release")

    stale_run = copy.deepcopy(complete)
    stale_run["stability"]["stale_run_results_detected"] = True
    if not any(
        "stale_run_results_detected" in issue
        for issue in validate(policy, stale_run)
    ):
        raise AssertionError("old Run leakage must block release")

    require_text(
        "scripts/package_vehicle_release.ps1",
        [
            "validate_vehicle_release_evidence.py",
            "audit_model_delivery.py",
            "Get-FileHash",
            "Compress-Archive",
            "status --porcelain",
        ],
    )
    require_text(
        "scripts/new_vehicle_release_evidence.ps1",
        [
            "vehicle_release_evidence.pending.v1.json",
            "Get-FileHash",
            "audit_model_delivery.py",
            "validate_vehicle_release_evidence.py",
        ],
    )
    print("PASS: M5 release evidence fails closed and packaging is traceable")


if __name__ == "__main__":
    main()
