#!/usr/bin/env python3
"""Validate M5 vehicle release evidence against the frozen acceptance policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
GATE_STATUSES = {"pending", "passed", "failed"}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and ":" not in path.parts[0]
        and all(part not in ("", ".") for part in path.parts)
    )


def read_mapping(
    value: object, name: str, issues: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(f"{name} must be an object")
        return {}
    return value


def gate_passed(
    gate: dict[str, Any], name: str, issues: list[str]
) -> None:
    status = gate.get("status")
    if status not in GATE_STATUSES:
        issues.append(f"{name}.status is invalid")
    elif status != "passed":
        issues.append(f"{name}.status={status!r} is not passed")
    evidence_path = gate.get("evidence_path")
    if not safe_relative_path(evidence_path):
        issues.append(f"{name}.evidence_path must be a safe project-relative path")


def number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def validate(policy: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    artifact_paths: set[str] = set()
    referenced_evidence_paths: set[str] = set()
    if evidence.get("schema_version") != "1.0":
        issues.append("schema_version must be '1.0'")
    if not isinstance(evidence.get("release_id"), str) or not SAFE_ID.fullmatch(
        evidence["release_id"]
    ):
        issues.append("release_id must be a safe identifier")
    if evidence.get("status") != "passed":
        issues.append("release status must be 'passed'")

    source = read_mapping(evidence.get("source"), "source", issues)
    commit = source.get("commit")
    if not isinstance(commit, str) or not SHA40.fullmatch(commit):
        issues.append("source.commit must be a lowercase 40-character Git SHA")
    if source.get("dirty") is not False:
        issues.append("source.dirty must be false")

    versions = read_mapping(evidence.get("versions"), "versions", issues)
    for field in ("dataset", "labels", "model_registry", "vehicle_config"):
        if not isinstance(versions.get(field), str) or not versions[field]:
            issues.append(f"versions.{field} must be frozen")
    dataset_manifest = versions.get("dataset_manifest")
    if not safe_relative_path(dataset_manifest):
        issues.append("versions.dataset_manifest must be a safe project-relative path")
    if versions.get("database_schema") != 4:
        issues.append("versions.database_schema must be 4")

    artifacts = evidence.get("artifact_hashes")
    if not isinstance(artifacts, list) or not artifacts:
        issues.append("artifact_hashes must contain the release inputs")
    else:
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                issues.append(f"artifact_hashes[{index}] must be an object")
                continue
            path = artifact.get("path")
            digest = artifact.get("sha256")
            if not safe_relative_path(path):
                issues.append(f"artifact_hashes[{index}].path is unsafe")
            elif path in artifact_paths:
                issues.append(f"artifact_hashes contains duplicate path {path}")
            else:
                artifact_paths.add(path)
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                issues.append(
                    f"artifact_hashes[{index}].sha256 must be lowercase SHA256"
                )
    if safe_relative_path(dataset_manifest) and dataset_manifest not in artifact_paths:
        issues.append("versions.dataset_manifest must be present in artifact_hashes")

    gate_passed(
        read_mapping(evidence.get("model_delivery"), "model_delivery", issues),
        "model_delivery",
        issues,
    )
    model_path = evidence.get("model_delivery", {}).get("evidence_path")
    if safe_relative_path(model_path):
        referenced_evidence_paths.add(model_path)

    performance_policy = read_mapping(
        policy.get("performance"), "policy.performance", issues
    )
    performance = read_mapping(
        evidence.get("performance"), "performance", issues
    )
    if performance.get("status") != "passed":
        issues.append("performance.status must be 'passed'")
    duration = number(performance.get("duration_minutes"))
    minimum_duration = number(
        performance_policy.get("minimum_duration_minutes")
    )
    if (
        duration is None
        or minimum_duration is None
        or duration < minimum_duration
    ):
        issues.append("performance duration is below the policy minimum")
    if performance.get("queue_growth_detected") is not False:
        issues.append("performance queue growth must not be detected")
    cameras = performance.get("cameras")
    required_count = performance_policy.get("camera_count")
    if not isinstance(cameras, list) or len(cameras) != required_count:
        issues.append(
            f"performance must contain exactly {required_count} camera results"
        )
    else:
        camera_ids: set[str] = set()
        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                issues.append(f"performance.cameras[{index}] must be an object")
                continue
            camera_id = camera.get("camera_id")
            if (
                not isinstance(camera_id, str)
                or not SAFE_ID.fullmatch(camera_id)
                or camera_id in camera_ids
            ):
                issues.append(
                    f"performance.cameras[{index}].camera_id is invalid or duplicate"
                )
            else:
                camera_ids.add(camera_id)
            if camera.get("width") != performance_policy.get("width") or camera.get(
                "height"
            ) != performance_policy.get("height"):
                issues.append(
                    f"performance.cameras[{index}] resolution is not "
                    f"{performance_policy.get('width')}x"
                    f"{performance_policy.get('height')}"
                )
            fps = number(camera.get("detection_fps"))
            minimum_fps = number(
                performance_policy.get("minimum_detection_fps_per_camera")
            )
            if fps is None or minimum_fps is None or fps < minimum_fps:
                issues.append(
                    f"performance.cameras[{index}].detection_fps is below "
                    f"{minimum_fps}"
                )
    if not safe_relative_path(performance.get("evidence_path")):
        issues.append("performance.evidence_path is required")
    else:
        referenced_evidence_paths.add(performance["evidence_path"])

    latency_policy = read_mapping(policy.get("latency"), "policy.latency", issues)
    latency = read_mapping(evidence.get("latency"), "latency", issues)
    if latency.get("status") != "passed":
        issues.append("latency.status must be 'passed'")
    for field, policy_field in (
        ("attribute_stable_p95_ms", "attribute_stable_p95_ms_max"),
        ("event_persist_p95_ms", "event_persist_p95_ms_max"),
    ):
        observed = number(latency.get(field))
        maximum = number(latency_policy.get(policy_field))
        if observed is None or maximum is None or observed > maximum:
            issues.append(f"latency.{field} exceeds {maximum}")
    if not safe_relative_path(latency.get("evidence_path")):
        issues.append("latency.evidence_path is required")
    else:
        referenced_evidence_paths.add(latency["evidence_path"])

    stability_policy = read_mapping(
        policy.get("stability"), "policy.stability", issues
    )
    stability = read_mapping(evidence.get("stability"), "stability", issues)
    if stability.get("status") != "passed":
        issues.append("stability.status must be 'passed'")
    hours = number(stability.get("duration_hours"))
    minimum_hours = number(stability_policy.get("minimum_duration_hours"))
    if hours is None or minimum_hours is None or hours < minimum_hours:
        issues.append("stability duration is below the policy minimum")
    if stability.get("crashes") != stability_policy.get("maximum_crashes"):
        issues.append("stability crashes exceed the policy maximum")
    for field in (
        "gpu_memory_growth_detected",
        "unbounded_reconnect_detected",
        "stale_run_results_detected",
    ):
        if stability.get(field) is not False:
            issues.append(f"stability.{field} must be false")
    if not safe_relative_path(stability.get("evidence_path")):
        issues.append("stability.evidence_path is required")
    else:
        referenced_evidence_paths.add(stability["evidence_path"])

    idempotency_policy = read_mapping(
        policy.get("idempotency"), "policy.idempotency", issues
    )
    idempotency = read_mapping(
        evidence.get("idempotency"), "idempotency", issues
    )
    if idempotency.get("status") != "passed":
        issues.append("idempotency.status must be 'passed'")
    if idempotency.get("duplicate_final_events") != idempotency_policy.get(
        "maximum_duplicate_final_events"
    ):
        issues.append("duplicate final events exceed the policy maximum")
    if not safe_relative_path(idempotency.get("evidence_path")):
        issues.append("idempotency.evidence_path is required")
    else:
        referenced_evidence_paths.add(idempotency["evidence_path"])

    fault_drills = read_mapping(
        evidence.get("fault_drills"), "fault_drills", issues
    )
    for drill in policy.get("required_fault_drills", []):
        drill_gate = read_mapping(
            fault_drills.get(drill), f"fault_drills.{drill}", issues
        )
        gate_passed(
            drill_gate,
            f"fault_drills.{drill}",
            issues,
        )
        drill_path = drill_gate.get("evidence_path")
        if safe_relative_path(drill_path):
            referenced_evidence_paths.add(drill_path)

    rollback = read_mapping(evidence.get("rollback"), "rollback", issues)
    for check in policy.get("required_rollback_checks", []):
        if rollback.get(check) != "passed":
            issues.append(f"rollback.{check} must be 'passed'")
    if not safe_relative_path(rollback.get("evidence_path")):
        issues.append("rollback.evidence_path is required")
    else:
        referenced_evidence_paths.add(rollback["evidence_path"])
    for path in sorted(referenced_evidence_paths - artifact_paths):
        issues.append(f"referenced evidence is missing from artifact_hashes: {path}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/vehicle_release_acceptance.v1.json"),
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="report blocking gates but return success for planning records",
    )
    args = parser.parse_args()
    issues = validate(load_object(args.policy), load_object(args.evidence))
    if issues:
        for issue in issues:
            print(f"BLOCKED: {issue}")
        print(f"SUMMARY: vehicle release is blocked by {len(issues)} gate(s)")
        return 0 if args.allow_pending else 1
    print("PASS: vehicle release evidence satisfies every frozen M5 gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
