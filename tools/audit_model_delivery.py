#!/usr/bin/env python3
"""Audit real model payloads, hashes, provenance, and delivery readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


READY_STATUSES = {"onnx_validated", "engine_validated", "deployed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_project_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    relative = PurePosixPath(value.removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
        return None
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def audit(registry: dict, project_root: Path) -> list[str]:
    issues: list[str] = []
    for artifact in registry.get("artifacts", []):
        artifact_id = artifact.get("artifact_id", "<unknown>")
        status = artifact.get("delivery_status")
        if status not in READY_STATUSES:
            issues.append(f"{artifact_id}: delivery_status={status!r} is not ready")
        files = artifact.get("files", {})
        for kind in ("onnx", "engine"):
            path = safe_project_path(project_root, files.get(f"{kind}_path"))
            expected = files.get(f"{kind}_sha256")
            if path is None:
                issues.append(f"{artifact_id}: unsafe or missing {kind}_path")
                continue
            if not path.is_file():
                issues.append(f"{artifact_id}: missing {kind} file {path}")
                continue
            actual = sha256(path)
            if not isinstance(expected, str):
                issues.append(f"{artifact_id}: {kind}_sha256 is not recorded")
            elif actual != expected:
                issues.append(
                    f"{artifact_id}: {kind} SHA256 mismatch {actual} != {expected}"
                )
        provenance = artifact.get("provenance", {})
        for field in (
            "dataset_version",
            "training_run_id",
            "code_commit",
            "model_card_path",
            "metrics_path",
        ):
            if provenance.get(field) in (None, ""):
                issues.append(f"{artifact_id}: missing provenance.{field}")
        for field in ("model_card_path", "metrics_path"):
            value = provenance.get(field)
            if value in (None, ""):
                continue
            path = safe_project_path(project_root, value)
            if path is None or not path.is_file():
                issues.append(f"{artifact_id}: provenance.{field} does not resolve")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("models/manifests/model_registry.v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--allow-planned",
        action="store_true",
        help="report missing delivery evidence without failing",
    )
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    issues = audit(registry, args.project_root.resolve())
    if issues:
        for issue in issues:
            print(f"BLOCKED: {issue}")
        print(f"SUMMARY: model delivery is blocked by {len(issues)} issue(s)")
        return 0 if args.allow_planned else 1
    print("PASS: all model payloads, hashes, provenance, and evidence are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
