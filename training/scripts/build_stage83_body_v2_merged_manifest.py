#!/usr/bin/env python3
"""Merge Stage72 CCTV body data with passing Stage82 MIO train-only truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_stage70_specialist_manifests import HammingBKTree


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def group_key(row: dict[str, str]) -> str:
    for field in ("track_group", "video_id", "source_group"):
        value = str(row.get(field, "")).strip()
        if value and value.lower() != "unknown":
            return f"{field}:{value}"
    return ""


def normalize_stage82_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a passing research-only row to the loader's approved contract."""
    if row.get("review_status", "").strip().lower() != "approved_research_only":
        raise ValueError("Stage82 row lost approved_research_only provenance")
    if not truthy(row.get("license_train_eligible")):
        raise ValueError("Stage82 row is not license-train-eligible")
    if row.get("source_license", "").strip().upper() != "CC-BY-NC-SA-4.0":
        raise ValueError("Stage82 row has an unexpected license")
    output = dict(row)
    output["review_status"] = "approved"
    output["research_only"] = "true"
    output["deployment_eligible"] = "false"
    output["stage83_license_scope"] = "CC-BY-NC-SA-4.0_research_only_non_deployable"
    if output.get("body_type", "").strip().lower() == "pickup":
        output["coarse_body_family"] = ""
        output["stage83_coarse_family_normalization"] = "pickup_exact_not_truck_family"
    return output


def pickup_coarse_conflict_count(rows: list[dict[str, str]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("stage83_origin") == "stage82_mio"
        and row.get("body_type", "").strip().lower() == "pickup"
        and bool(row.get("coarse_body_family", "").strip())
    )


def build_report(
    base_path: Path,
    addition_path: Path,
    addition_report_path: Path,
    labels_path: Path,
    expected_base_sha256: str,
    expected_addition_sha256: str,
    expected_addition_report_sha256: str,
    expected_labels_sha256: str,
    near_duplicate_hamming: int,
) -> dict[str, object]:
    actual = {
        "base": sha256(base_path),
        "addition": sha256(addition_path),
        "addition_report": sha256(addition_report_path),
        "labels": sha256(labels_path),
    }
    expected = {
        "base": expected_base_sha256.lower(),
        "addition": expected_addition_sha256.lower(),
        "addition_report": expected_addition_report_sha256.lower(),
        "labels": expected_labels_sha256.lower(),
    }
    mismatches = [name for name in expected if actual[name].lower() != expected[name]]
    if mismatches:
        raise RuntimeError(f"immutable input SHA256 mismatch: {mismatches}")

    base_fields, base_rows = read_manifest(base_path)
    addition_fields, addition_rows = read_manifest(addition_path)
    addition_report = json.loads(addition_report_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if addition_report.get("status") != "pass":
        failures.append("Stage82 report is not pass")
    policy = addition_report.get("policy", {})
    if policy.get("official_test_image_payloads_read") != 0:
        failures.append("Stage82 official test image payload policy failed")
    if policy.get("validation_rows_created") != 0 or policy.get("test_rows_created") != 0:
        failures.append("Stage82 created a held-out row")
    dedup_ref = addition_report.get("successor_dedup_reference", {})
    if str(dedup_ref.get("sha256", "")).lower() != actual["base"].lower():
        failures.append("Stage82 dedup reference does not bind the Stage72 base")
    if int(dedup_ref.get("near_duplicate_hamming", -1)) < near_duplicate_hamming:
        failures.append("Stage82 cross-source near-duplicate radius is too small")

    allowed_body = set(labels.get("body_types", []))
    if any(row.get("split", "").strip().lower() != "train" for row in addition_rows):
        failures.append("Stage82 contains a non-train row")
    bad_labels = sorted({row.get("body_type", "") for row in addition_rows if row.get("body_type", "") not in allowed_body})
    if bad_labels:
        failures.append(f"Stage82 body labels outside v2 contract: {bad_labels}")
    if any(truthy(row.get("color_supervised")) for row in addition_rows):
        failures.append("Stage82 unexpectedly contains color supervision")
    normalized_addition: list[dict[str, str]] = []
    for index, row in enumerate(addition_rows, start=2):
        try:
            normalized_addition.append(normalize_stage82_row(row))
        except ValueError as exc:
            failures.append(f"Stage82 line {index}: {exc}")

    validation = [row for row in base_rows if row.get("split", "").strip().lower() == "validation"]
    validation_exact = {content_hash(row) for row in validation if content_hash(row)}
    validation_tree = HammingBKTree()
    for row in validation:
        value = perceptual_hash(row)
        if len(value) == 16:
            validation_tree.add(int(value, 16), row)
    validation_groups = {group_key(row) for row in validation if group_key(row)}
    exact_leaks = 0
    near_leaks = 0
    group_leaks = 0
    for row in addition_rows:
        digest = content_hash(row)
        if digest and digest in validation_exact:
            exact_leaks += 1
        value = perceptual_hash(row)
        if len(value) == 16 and validation_tree.find(int(value, 16), near_duplicate_hamming) is not None:
            near_leaks += 1
        group = group_key(row)
        if group and group in validation_groups:
            group_leaks += 1
    if exact_leaks:
        failures.append(f"Stage82-to-validation exact leaks: {exact_leaks}")
    if near_leaks:
        failures.append(f"Stage82-to-validation near leaks: {near_leaks}")
    if group_leaks:
        failures.append(f"Stage82-to-validation group leaks: {group_leaks}")

    merged: list[dict[str, str]] = []
    for row in base_rows:
        output = dict(row)
        output["stage83_origin"] = "stage72"
        merged.append(output)
    for row in normalized_addition:
        output = dict(row)
        output["stage83_origin"] = "stage82_mio"
        merged.append(output)
    frozen_rows = sum(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in merged
    )
    test_rows = sum(row.get("split", "").strip().lower() == "test" for row in merged)
    if frozen_rows:
        failures.append(f"frozen markers found: {frozen_rows}")
    if test_rows:
        failures.append(f"test rows found: {test_rows}")

    loader_usable_supervised_train = sum(
        row.get("split", "").strip().lower() == "train"
        and row.get("review_status", "").strip().lower() == "approved"
        and truthy(row.get("body_type_supervised"))
        and row.get("body_type", "") in allowed_body
        for row in merged
    )
    stage82_loader_usable = sum(
        row.get("stage83_origin") == "stage82_mio"
        and row.get("review_status", "").strip().lower() == "approved"
        and truthy(row.get("body_type_supervised"))
        and row.get("body_type", "") in allowed_body
        for row in merged
    )
    pickup_coarse_conflicts = pickup_coarse_conflict_count(merged)
    pickup_coarse_family_cleared = sum(
        row.get("stage83_coarse_family_normalization") == "pickup_exact_not_truck_family"
        for row in merged
    )
    if stage82_loader_usable != len(addition_rows):
        failures.append(
            f"Stage82 loader-usable rows: {stage82_loader_usable} != {len(addition_rows)}"
        )
    if pickup_coarse_conflicts:
        failures.append(f"pickup coarse-truck loss conflicts: {pickup_coarse_conflicts}")

    split_counts = Counter(row.get("split", "") for row in merged)
    body_counts = Counter(
        f"{row.get('split', '')}:{row.get('body_type', '')}"
        for row in merged
        if truthy(row.get("body_type_supervised"))
    )
    return {
        "report": {
            "schema_version": "stage83-body-v2-merged-manifest-v1",
            "status": "pass" if not failures else "fail",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "stage72_manifest": str(base_path.resolve()), "stage72_sha256": actual["base"],
                "stage82_manifest": str(addition_path.resolve()), "stage82_sha256": actual["addition"],
                "stage82_report": str(addition_report_path.resolve()),
                "stage82_report_sha256": actual["addition_report"],
                "labels": str(labels_path.resolve()), "labels_sha256": actual["labels"],
            },
            "output": {
                "rows": len(merged), "splits": dict(split_counts),
                "body_truth_counts": dict(sorted(body_counts.items())),
                "stage72_rows": len(base_rows), "stage82_rows": len(addition_rows),
                "loader_usable_supervised_train_rows": loader_usable_supervised_train,
                "stage82_loader_usable_rows": stage82_loader_usable,
                "stage82_pickup_coarse_family_cleared": pickup_coarse_family_cleared,
            },
            "integrity": {
                "stage82_to_validation_exact_leaks": exact_leaks,
                "stage82_to_validation_near_leaks": near_leaks,
                "stage82_to_validation_group_leaks": group_leaks,
                "test_rows": test_rows, "frozen_markers": frozen_rows,
                "near_duplicate_hamming": near_duplicate_hamming,
            },
            "policy": {
                "stage82_is_train_only": True,
                "stage72_validation_preserved": True,
                "stage82_review_status_normalized_for_loader": True,
                "stage82_research_only_scope_preserved": True,
                "pickup_exact_truth_excluded_from_coarse_truck_loss": True,
                "test_accessed": False,
                "frozen_video_used": False,
                "production_model_modified": False,
                "deployment_performed": False,
                "eligibility": "research-only_non-deployable",
            },
            "failures": failures,
        },
        "rows": merged,
        "fields": list(dict.fromkeys([
            *base_fields, *addition_fields, "research_only", "deployment_eligible",
            "stage83_license_scope", "stage83_coarse_family_normalization", "stage83_origin",
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--addition-manifest", type=Path, required=True)
    parser.add_argument("--expected-addition-sha256", required=True)
    parser.add_argument("--addition-report", type=Path, required=True)
    parser.add_argument("--expected-addition-report-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage83 evidence")
    result = build_report(
        args.base_manifest, args.addition_manifest, args.addition_report, args.labels,
        args.expected_base_sha256, args.expected_addition_sha256,
        args.expected_addition_report_sha256, args.expected_labels_sha256,
        args.near_duplicate_hamming,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["report"]["status"] != "pass":
        raise SystemExit(1)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    result["report"]["output"]["manifest"] = str(args.output_manifest.resolve())
    result["report"]["output"]["manifest_sha256"] = sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": result["report"]["output"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
