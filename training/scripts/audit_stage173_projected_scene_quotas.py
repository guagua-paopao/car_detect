#!/usr/bin/env python3
"""Project scene quotas for Stage167 plus sealed Stage172 without creating a training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def signals(row: dict[str, str]) -> tuple[bool, bool, bool]:
    lighting = str(row.get("lighting") or "").strip().lower()
    occlusion = str(row.get("occlusion_level") or "").strip().lower()
    night = truthy(row.get("night")) or truthy(row.get("low_light")) or lighting in {
        "night", "low_light", "night_source_truth",
    }
    small = truthy(row.get("small_target")) or str(row.get("vehicle_size") or "").strip().lower() == "small"
    occluded = (
        truthy(row.get("occluded")) or truthy(row.get("truncated"))
        or occlusion not in {"", "none", "clear", "unknown"}
    )
    return night, small, occluded


def summarize(path: Path) -> dict[str, dict[str, object]]:
    exact: Counter[str] = Counter()
    body_signal: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split") or "train").strip().lower() != "train":
                continue
            exact_truth = truthy(row.get("body_type_supervised"))
            coarse_truth = str(row.get("coarse_body_family") or "").strip().lower() in {"car", "truck"}
            night, small, occluded = signals(row)
            for counter, admitted in ((exact, exact_truth), (body_signal, exact_truth or coarse_truth)):
                if not admitted:
                    continue
                counter["rows"] += 1
                counter["night"] += night
                counter["small"] += small
                counter["occlusion"] += occluded
    def finish(counter: Counter[str]) -> dict[str, object]:
        rows = counter["rows"]
        return {
            "rows": rows,
            "night": counter["night"], "night_fraction": counter["night"] / rows if rows else 0.0,
            "small": counter["small"], "small_fraction": counter["small"] / rows if rows else 0.0,
            "occlusion": counter["occlusion"], "occlusion_fraction": counter["occlusion"] / rows if rows else 0.0,
        }
    return {"exact": finish(exact), "exact_or_coarse": finish(body_signal)}


def combine(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    rows = int(left["rows"]) + int(right["rows"])
    result: dict[str, object] = {"rows": rows}
    for key in ("night", "small", "occlusion"):
        count = int(left[key]) + int(right[key])
        result[key] = count
        result[f"{key}_fraction"] = count / rows if rows else 0.0
    return result


def additional_rows_needed(rows: int, count: int, target: float) -> int:
    if rows <= 0 or count / rows >= target:
        return 0
    return max(0, math.ceil((target * rows - count) / (1.0 - target)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage167", type=Path, required=True)
    parser.add_argument("--expected-stage167-sha256", required=True)
    parser.add_argument("--stage172", type=Path, required=True)
    parser.add_argument("--expected-stage172-sha256", required=True)
    parser.add_argument("--stage172-audit", type=Path, required=True)
    parser.add_argument("--expected-stage172-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {
        "stage167": args.stage167.resolve(),
        "stage172": args.stage172.resolve(),
        "stage172_audit": args.stage172_audit.resolve(),
    }
    actual = {key: sha256_file(path) for key, path in inputs.items()}
    expected = {
        "stage167": args.expected_stage167_sha256.lower(),
        "stage172": args.expected_stage172_sha256.lower(),
        "stage172_audit": args.expected_stage172_audit_sha256.lower(),
    }
    if actual != expected:
        raise RuntimeError(f"pinned input mismatch: {actual}")
    source_audit = json.loads(inputs["stage172_audit"].read_text(encoding="utf-8"))
    if source_audit.get("status") != "pass" or source_audit.get("failures"):
        raise RuntimeError("Stage172 independent audit did not pass")
    stage167 = summarize(inputs["stage167"])
    stage172 = summarize(inputs["stage172"])
    combined = {
        scope: combine(stage167[scope], stage172[scope])
        for scope in ("exact", "exact_or_coarse")
    }
    targets = {"night": 0.30, "small": 0.20, "occlusion": 0.15}
    gates: dict[str, dict[str, object]] = {}
    for scope, summary in combined.items():
        gates[scope] = {}
        for key, target in targets.items():
            fraction = float(summary[f"{key}_fraction"])
            gates[scope][key] = {
                "target": target,
                "fraction": fraction,
                "passes": fraction >= target,
                "additional_all_positive_rows_needed": additional_rows_needed(
                    int(summary["rows"]), int(summary[key]), target,
                ),
            }
    report = {
        "schema_version": "stage173-projected-scene-quotas-v1",
        "status": "pass_projection_night_gap_remains",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            key: {"path": str(path), "sha256": actual[key]} for key, path in inputs.items()
        },
        "stage167": stage167,
        "stage172": stage172,
        "combined_projection": combined,
        "quota_gates": gates,
        "interpretation": (
            "Stage172 closes the projected small and occlusion quotas only when legitimate coarse family truth is counted; "
            "the 30% night quota remains unmet in both exact-only and exact-or-coarse views. Sampling weights cannot replace missing unique rows."
        ),
        "policy": {
            "merged_manifest_created": False,
            "training_started": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "combined_exact_or_coarse": combined["exact_or_coarse"],
        "night_rows_needed": gates["exact_or_coarse"]["night"]["additional_all_positive_rows_needed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
