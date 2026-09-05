#!/usr/bin/env python3
"""Create a runtime-safe Stage74 joint manifest without changing its labels or split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def projection_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = [str(row.get(field, "")) for field in fields]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_runtime_path(
    row: dict[str, str],
    *,
    base_parent: Path,
    ua_root: Path,
    bmd_root: Path,
) -> tuple[Path, str]:
    source = row.get("stage74_joint_source", "")
    raw = Path(row.get("image_path", ""))
    if not str(raw):
        raise RuntimeError("manifest row has an empty image_path")
    if raw.is_absolute():
        return raw.resolve(), "already_absolute"
    if source == "ua":
        return (ua_root / raw).resolve(), "ua_absolute_normalized"
    if source == "bmd":
        return (bmd_root / raw).resolve(), "bmd_absolute_normalized"
    if source:
        raise RuntimeError(f"unsupported Stage74 joint source: {source}")
    return (base_parent / raw).resolve(), "base_relative_preserved"


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--expected-input-report-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--ua-root", type=Path, required=True)
    parser.add_argument("--bmd-root", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage74 runtime evidence: {path}")
    pinned = (
        (args.input_manifest, args.expected_input_manifest_sha256),
        (args.input_report, args.expected_input_report_sha256),
        (args.base_manifest, args.expected_base_manifest_sha256),
    )
    for path, expected in pinned:
        if not path.is_file() or sha256(path).lower() != expected.lower():
            raise RuntimeError(f"immutable Stage74 input mismatch: {path}")

    input_report = json.loads(args.input_report.read_text(encoding="utf-8"))
    if input_report.get("status") != "pass_joint_research_manifest_available":
        raise RuntimeError("input Stage74 joint report did not pass")
    if input_report.get("eligibility") != "research-only_non-deployable":
        raise RuntimeError("input Stage74 report lost research-only isolation")
    if input_report.get("output_manifest_sha256", "").lower() != sha256(args.input_manifest):
        raise RuntimeError("input Stage74 report does not bind its manifest")

    fields, rows = load_csv(args.input_manifest)
    base_fields, base_rows = load_csv(args.base_manifest)
    if len(rows) != int(input_report.get("joint_rows", -1)):
        raise RuntimeError("input joint row count changed")
    if len(base_rows) != int(input_report.get("base_rows", -1)):
        raise RuntimeError("base row count changed")
    if args.output_manifest.parent.resolve() != args.base_manifest.parent.resolve():
        raise RuntimeError("runtime manifest must share the base manifest parent so base-relative paths retain meaning")

    safety_root = args.datasets_safety_root.resolve()
    ua_root = args.ua_root.resolve()
    bmd_root = args.bmd_root.resolve()
    base_parent = args.base_manifest.parent.resolve()
    for root in (safety_root, ua_root, bmd_root, base_parent):
        if not root.is_dir():
            raise RuntimeError(f"required Stage74 runtime root is missing: {root}")

    output_rows: list[dict[str, str]] = []
    path_modes: Counter[str] = Counter()
    for original in rows:
        if original.get("split") not in {"train", "validation"}:
            raise RuntimeError("runtime manifest contains a forbidden split")
        if "test" in original.get("image_path", "").lower():
            raise RuntimeError("runtime manifest contains a test-like path")
        resolved, mode = resolve_runtime_path(
            original,
            base_parent=base_parent,
            ua_root=ua_root,
            bmd_root=bmd_root,
        )
        try:
            resolved.relative_to(safety_root)
        except ValueError as error:
            raise RuntimeError(f"runtime image escapes datasets safety root: {resolved}") from error
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise RuntimeError(f"runtime image is missing or empty: {resolved}")
        row = dict(original)
        if mode in {"ua_absolute_normalized", "bmd_absolute_normalized"}:
            row["image_path"] = str(resolved)
        output_rows.append(row)
        path_modes[mode] += 1

    input_validation = [row for row in rows if row.get("split") == "validation"]
    output_validation = [row for row in output_rows if row.get("split") == "validation"]
    before = projection_digest(input_validation, base_fields)
    after = projection_digest(output_validation, base_fields)
    if before != after or before != input_report.get("validation_projection_sha256_after"):
        raise RuntimeError("Stage74 validation projection changed during runtime normalization")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.output_manifest, fields, output_rows)
    output_manifest_sha = sha256(args.output_manifest)
    report = {
        **input_report,
        "schema_version": "stage74-joint-color-runtime-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_joint_research_manifest_available",
        "eligibility": "research-only_non-deployable",
        "runtime_normalization": {
            "status": "pass_all_paths_resolved_nonempty_within_safety_root",
            "input_manifest": str(args.input_manifest.resolve()),
            "input_manifest_sha256": sha256(args.input_manifest),
            "input_report": str(args.input_report.resolve()),
            "input_report_sha256": sha256(args.input_report),
            "base_manifest": str(args.base_manifest.resolve()),
            "base_manifest_sha256": sha256(args.base_manifest),
            "path_modes": dict(sorted(path_modes.items())),
            "rows_checked": len(output_rows),
            "validation_projection_unchanged": True,
        },
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": output_manifest_sha,
        "policy": {
            **input_report.get("policy", {}),
            "runtime_paths_audited": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only this path-normalized manifest may retry the isolated Stage74 candidate; V1 failure evidence remains immutable",
    }
    atomic_write_json(args.output_report, report)
    print(json.dumps({
        "status": report["status"],
        "rows_checked": len(output_rows),
        "path_modes": dict(sorted(path_modes.items())),
        "manifest_sha256": output_manifest_sha,
        "test_accessed": False,
        "frozen_video_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
