#!/usr/bin/env python3
"""Add official two-axle toll-CCTV truth to the truck subtype manifest.

Stage93 deliberately treated iNATRC class 1 as generic ``truck``.  Error
analysis on the independent VFG validation set showed that the target
``light_truck`` class is a two-axle commercial truck, while three-plus-axle
vehicles are ``heavy_truck``.  This builder therefore applies the project's
explicit axle-based subtype contract to the publisher's official class-2
two-axle rows.  It never reads publisher validation/test data or VFG images.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_stage93_truck_subtype_manifest import (
    FROZEN_MARKERS,
    deduplicate_validation_first,
    post_split_leaks,
    read_manifest,
    readable_image,
    resolve_image,
    sha256,
    truthy,
)


FINE_LABELS = {"light_truck", "heavy_truck"}
TWO_AXLE_SOURCE_LABEL = "two_axle_truck"


def validate_source_evidence(evidence: dict[str, object]) -> None:
    normalized_license = str(evidence.get("license", "")).upper().replace("-", " ")
    if normalized_license != "CC BY 4.0":
        raise RuntimeError("iNATRC source evidence lost CC-BY-4.0 license")
    classes = evidence.get("class_contract")
    if not isinstance(classes, dict) or "two axles" not in str(classes.get("1", "")).lower():
        raise RuntimeError("official class 1 is not evidenced as a two-axle truck")
    policy = evidence.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("iNATRC source evidence lacks dataset scope")


def eligible_two_axle(row: dict[str, str]) -> bool:
    return (
        row.get("split", "").strip().lower() == "train"
        and row.get("body_type", "").strip().lower() == "truck"
        and row.get("source_label", "").strip().lower() == TWO_AXLE_SOURCE_LABEL
        and truthy(row.get("body_type_supervised"))
        and truthy(row.get("license_train_eligible"))
    )


def normalize_two_axle(
    row: dict[str, str], *, manifest: Path, safety_root: Path
) -> dict[str, str]:
    output = dict(row)
    output["image_path"] = str(resolve_image(manifest, safety_root, row.get("image_path", "")))
    output["body_type"] = "light_truck"
    output["body_type_supervised"] = "true"
    output["color"] = "unknown"
    output["color_supervised"] = "false"
    output["review_status"] = "approved"
    output["sample_weight"] = "1.500"
    output["research_only"] = "false"
    output["deployment_eligible"] = "true"
    output["stage98_origin"] = "stage91_inatrc_two_axle_train"
    output["stage98_original_body_type"] = row.get("body_type", "")
    output["stage98_fine_truth_source"] = "official_class_2_two_axle_toll_cctv"
    output["stage98_mapping_contract"] = "two_axle_to_light_truck;three_plus_axle_to_heavy_truck"
    return output


def build(args: argparse.Namespace) -> dict[str, object]:
    inputs = {
        "stage93": args.stage93_manifest,
        "stage91": args.stage91_manifest,
        "stage91_report": args.stage91_report,
        "source_evidence": args.source_evidence,
        "labels": args.labels,
    }
    expected = {
        "stage93": args.expected_stage93_sha256.lower(),
        "stage91": args.expected_stage91_sha256.lower(),
        "stage91_report": args.expected_stage91_report_sha256.lower(),
        "source_evidence": args.expected_source_evidence_sha256.lower(),
        "labels": args.expected_labels_sha256.lower(),
    }
    actual = {name: sha256(path) for name, path in inputs.items()}
    mismatches = [name for name in inputs if actual[name].lower() != expected[name]]
    if mismatches:
        raise RuntimeError(f"immutable input SHA256 mismatch: {mismatches}")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("body_types") != ["light_truck", "heavy_truck", "unknown"]:
        raise RuntimeError("truck subtype labels contract mismatch")
    if labels.get("colors") != ["unknown"]:
        raise RuntimeError("specialist color contract mismatch")
    evidence = json.loads(args.source_evidence.read_text(encoding="utf-8"))
    validate_source_evidence(evidence)
    stage91_report = json.loads(args.stage91_report.read_text(encoding="utf-8"))
    if stage91_report.get("status") != "pass":
        raise RuntimeError("pinned Stage91 crop report did not pass")
    policy = stage91_report.get("policy", {})
    if policy.get("source_validation_payloads_read", 0) or policy.get("source_test_payloads_read", 0):
        raise RuntimeError("Stage91 report indicates sealed split access")

    stage93_fields, stage93_rows = read_manifest(args.stage93_manifest)
    stage91_fields, stage91_rows = read_manifest(args.stage91_manifest)
    if any(row.get("split", "").strip().lower() == "test" for row in [*stage93_rows, *stage91_rows]):
        raise RuntimeError("test rows are forbidden in Stage98 inputs")
    if any(row.get("body_type", "").strip().lower() not in FINE_LABELS for row in stage93_rows):
        raise RuntimeError("pinned Stage93 manifest contains a non-fine label")

    additions = [
        normalize_two_axle(row, manifest=args.stage91_manifest, safety_root=args.safety_root)
        for row in stage91_rows
        if eligible_two_axle(row)
    ]
    frozen_rows = sum(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in additions
    )
    readable_additions: list[dict[str, str]] = []
    unreadable_additions: list[dict[str, str]] = []
    for row in additions:
        (readable_additions if readable_image(Path(row["image_path"])) else unreadable_additions).append(row)
    failures: list[str] = []
    if frozen_rows:
        failures.append(f"frozen markers found: {frozen_rows}")
    if unreadable_additions:
        failures.append(f"unreadable two-axle additions: {len(unreadable_additions)}")
    additions = readable_additions

    combined = [dict(row) for row in stage93_rows] + additions
    deduplicated, dedup = deduplicate_validation_first(combined, args.near_duplicate_hamming)
    leaks = post_split_leaks(deduplicated, args.near_duplicate_hamming)
    if any(leaks.values()):
        failures.append(f"post-split leaks: {leaks}")
    counts = Counter(f"{row['split']}:{row['body_type']}" for row in deduplicated)
    for label in sorted(FINE_LABELS):
        if counts[f"train:{label}"] < args.minimum_train_per_class:
            failures.append(f"train {label} below minimum")
        if counts[f"validation:{label}"] < args.minimum_validation_per_class:
            failures.append(f"validation {label} below minimum")
    if len(additions) < args.minimum_two_axle_rows:
        failures.append(f"two-axle additions {len(additions)} < {args.minimum_two_axle_rows}")
    if any(truthy(row.get("color_supervised")) or row.get("color") != "unknown" for row in deduplicated):
        failures.append("color supervision survived")

    report: dict[str, object] = {
        "schema_version": "stage98-two-axle-cctv-specialist-manifest-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": actual[name]}
            for name, path in inputs.items()
        },
        "mapping": {
            "publisher_class": "Class-2 truck with two axles",
            "source_label": TWO_AXLE_SOURCE_LABEL,
            "target_label": "light_truck",
            "contract": "two_axle_to_light_truck;three_plus_axle_to_heavy_truck",
            "rationale": "the target taxonomy defines light/heavy commercial trucks by axle class; exact independent validation remains mandatory",
        },
        "output": {
            "rows": len(deduplicated),
            "counts": dict(sorted(counts.items())),
            "eligible_two_axle_rows": len(additions),
            "unreadable_two_axle_rows": len(unreadable_additions),
        },
        "integrity": {
            "near_duplicate_hamming": args.near_duplicate_hamming,
            "dedup": dedup,
            "post_split_leaks": leaks,
            "frozen_markers": frozen_rows,
        },
        "policy": {
            "stage93_validation_preserved_and_protected_first": True,
            "stage91_train_only": True,
            "publisher_validation_payloads_read": 0,
            "publisher_test_payloads_read": 0,
            "vfg_validation_used_for_training": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable_until_independent_validation",
        },
        "failures": failures,
    }
    fields = list(dict.fromkeys([
        *stage93_fields,
        *stage91_fields,
        "stage98_origin",
        "stage98_original_body_type",
        "stage98_fine_truth_source",
        "stage98_mapping_contract",
    ]))
    return {"report": report, "rows": deduplicated, "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage93-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage93-sha256", required=True)
    parser.add_argument("--stage91-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage91-sha256", required=True)
    parser.add_argument("--stage91-report", type=Path, required=True)
    parser.add_argument("--expected-stage91-report-sha256", required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--expected-source-evidence-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--safety-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-train-per-class", type=int, default=10000)
    parser.add_argument("--minimum-validation-per-class", type=int, default=300)
    parser.add_argument("--minimum-two-axle-rows", type=int, default=4000)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage98 evidence")
    result = build(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["report"]["status"] != "pass":
        return 1
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    result["report"]["output"]["manifest"] = str(args.output_manifest.resolve())
    result["report"]["output"]["manifest_sha256"] = sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(result["rows"]), "counts": result["report"]["output"]["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
