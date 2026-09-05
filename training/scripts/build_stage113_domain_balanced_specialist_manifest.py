#!/usr/bin/env python3
"""Build a license-clean, road-domain-balanced truck specialist manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_stage93_truck_subtype_manifest import post_split_leaks, sha256, truthy


FINE_LABELS = {"light_truck", "heavy_truck"}
SOURCE_FACTORS = {
    "BMD-45-RAW": 2.00,
    "BMD-45": 2.00,
    "Open-Images-V7": 1.25,
    "UVH-26": 0.75,
    "InaTRC-v1": 0.25,
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def restricted(row: dict[str, str]) -> bool:
    license_name = str(row.get("source_license", "")).upper()
    return "NC" in license_name or truthy(row.get("research_only"))


def eligible_train(row: dict[str, str]) -> bool:
    return (
        row.get("split") == "train"
        and row.get("body_type") in FINE_LABELS
        and truthy(row.get("body_type_supervised"))
        and row.get("review_status") == "approved"
        and not restricted(row)
        and row.get("source_dataset") in SOURCE_FACTORS
    )


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    if any(row.get("split") == "test" for row in rows):
        raise RuntimeError("Stage113 input must not contain test rows")
    validation = [dict(row) for row in rows if row.get("split") == "validation"]
    candidates = [dict(row) for row in rows if eligible_train(row)]
    excluded = Counter()
    for row in rows:
        if row.get("split") != "train":
            continue
        if restricted(row):
            excluded["noncommercial_or_research_only"] += 1
        elif row.get("source_dataset") not in SOURCE_FACTORS:
            excluded["source_not_in_license_clean_balance_contract"] += 1
        elif not eligible_train(row):
            excluded["invalid_fine_supervision"] += 1
    raw_mass = Counter()
    for row in candidates:
        raw_mass[row["body_type"]] += SOURCE_FACTORS[row["source_dataset"]]
    if set(raw_mass) != FINE_LABELS or min(raw_mass.values()) <= 0:
        raise RuntimeError("both fine classes require eligible train mass")
    target_mass = max(raw_mass.values())
    class_multiplier = {label: target_mass / raw_mass[label] for label in FINE_LABELS}
    output_train: list[dict[str, str]] = []
    for row in candidates:
        output = dict(row)
        factor = SOURCE_FACTORS[row["source_dataset"]]
        weight = factor * class_multiplier[row["body_type"]]
        output["stage113_previous_sample_weight"] = row.get("sample_weight", "")
        output["sample_weight"] = f"{weight:.6f}"
        output["stage113_origin"] = "stage98_license_clean_source_balanced"
        output["stage113_source_factor"] = f"{factor:.6f}"
        output["stage113_class_multiplier"] = f"{class_multiplier[row['body_type']]:.6f}"
        output["stage113_domain_role"] = (
            "roadside_primary"
            if row["source_dataset"] in {"BMD-45-RAW", "BMD-45", "Open-Images-V7"}
            else "auxiliary"
        )
        output_train.append(output)
    output = [*validation, *output_train]
    train_counts = Counter(row["body_type"] for row in output_train)
    source_class = Counter(
        f"{row['source_dataset']}:{row['body_type']}" for row in output_train
    )
    weighted_mass = Counter()
    for row in output_train:
        weighted_mass[row["body_type"]] += float(row["sample_weight"])
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "validation_rows_preserved": len(validation),
        "train_rows": len(output_train),
        "train_counts": dict(sorted(train_counts.items())),
        "source_class_counts": dict(sorted(source_class.items())),
        "excluded": dict(sorted(excluded.items())),
        "raw_source_weight_mass": dict(sorted(raw_mass.items())),
        "class_multipliers": dict(sorted(class_multiplier.items())),
        "weighted_class_mass": dict(sorted(weighted_mass.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--expected-input-report-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage113 evidence")
    for path, expected, role in (
        (args.input_manifest, args.expected_input_manifest_sha256, "manifest"),
        (args.input_report, args.expected_input_report_sha256, "report"),
    ):
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    input_report = json.loads(args.input_report.read_text(encoding="utf-8"))
    if input_report.get("status") != "pass":
        raise RuntimeError("Stage98 input report did not pass")
    policy = input_report.get("policy", {})
    if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
        raise RuntimeError("Stage98 input policy lost test/frozen isolation")
    fields, rows = read_rows(args.input_manifest)
    frozen_rows = sum(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in rows
    )
    if frozen_rows:
        raise RuntimeError(f"frozen marker rows found: {frozen_rows}")
    output, summary = build(rows)
    if min(summary["train_counts"].values()) < 10000:
        raise RuntimeError("Stage113 license-clean class count below 10000")
    leaks = post_split_leaks(output, 4)
    if any(leaks.values()):
        raise RuntimeError(f"Stage113 cross-split leak: {leaks}")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=False)
    extra = (
        "stage113_previous_sample_weight",
        "stage113_origin",
        "stage113_source_factor",
        "stage113_class_multiplier",
        "stage113_domain_role",
    )
    output_fields = list(dict.fromkeys([*fields, *extra]))
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    report = {
        "schema_version": "stage113-domain-balanced-specialist-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "inputs": {
            "manifest_sha256": sha256(args.input_manifest),
            "report_sha256": sha256(args.input_report),
        },
        "output": {
            **summary,
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256(args.output_manifest),
        },
        "source_factors": SOURCE_FACTORS,
        "integrity": {"near_duplicate_hamming": 4, "post_split_leaks": leaks, "frozen_rows": 0},
        "policy": {
            "validation_rows_preserved_without_relabeling": True,
            "noncommercial_and_research_only_train_rows_excluded": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_manifest, args.output_report):
        Path(str(path) + ".sha256").write_text(
            f"{sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({"status": "pass", **summary, "leaks": leaks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
