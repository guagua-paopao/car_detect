#!/usr/bin/env python3
"""Merge curated Open Images hard supplements into the locked Stage50 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def rows_digest(rows: list[dict], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {field: row.get(field, "") for field in fields}
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def measure_blur(image_path: Path, threshold: float = 60.0) -> tuple[bool, float]:
    """Return a deterministic train-only blur proxy without inventing unknown metadata."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise RuntimeError(f"could not read supplement crop for blur audit: {image_path}")
    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return score < threshold, score


def validate_report(report_path: Path, manifest_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"input curation did not pass: {report_path}")
    if report.get("output_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(f"input curation manifest hash mismatch: {manifest_path}")
    policy = report.get("policy", {})
    if policy.get("train_only") is not True:
        raise RuntimeError(f"input curation is not train-only: {report_path}")
    if policy.get("validation_or_test_used") is not False:
        raise RuntimeError(f"input curation used validation/test: {report_path}")
    if policy.get("frozen_video_used") is not False:
        raise RuntimeError(f"input curation used frozen video: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--body-manifest", type=Path, required=True)
    parser.add_argument("--body-report", type=Path, required=True)
    parser.add_argument("--color-manifest", type=Path, required=True)
    parser.add_argument("--color-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage59 manifest evidence")

    body_report = validate_report(args.body_report, args.body_manifest)
    color_report = validate_report(args.color_report, args.color_manifest)
    base_fields, base_rows = load_csv(args.base_manifest)
    body_fields, body_rows = load_csv(args.body_manifest)
    color_fields, color_rows = load_csv(args.color_manifest)
    if any(row.get("split") != "train" for row in body_rows + color_rows):
        raise RuntimeError("supplement contains non-train rows")

    base_paths = {row["image_path"] for row in base_rows}
    base_groups = {
        value
        for row in base_rows
        for value in (row.get("camera_id", ""), row.get("video_id", ""), row.get("track_group", ""))
        if value
    }
    base_dhash = {row.get("dhash64", "") for row in base_rows if len(row.get("dhash64", "")) == 16}
    base_test_validation = [row for row in base_rows if row.get("split") in {"validation", "test"}]
    base_eval_digest = rows_digest(base_test_validation, base_fields)

    merged_supplement: dict[str, dict] = {}
    source_heads: dict[str, set[str]] = {}
    rejected = Counter()
    for head, rows in (("body", body_rows), ("color", color_rows)):
        for source in rows:
            path = source["image_path"]
            if path in base_paths:
                rejected["base_image_path_overlap"] += 1
                continue
            if not (args.dataset_root / path).is_file():
                rejected["missing_image"] += 1
                continue
            group_values = {
                source.get("camera_id", ""), source.get("video_id", ""), source.get("track_group", "")
            } - {""}
            if group_values & base_groups:
                rejected["base_group_overlap"] += 1
                continue
            row = merged_supplement.setdefault(path, dict(source))
            source_heads.setdefault(path, set()).add(head)
            if head == "body":
                row["body_type"] = source["body_type"]
                row["body_type_supervised"] = "true"
                for key, value in source.items():
                    if key.startswith("teacher_"):
                        row[key] = value
            else:
                row["color"] = source["color"]
                row["color_supervised"] = "true"
                row["color_review_status"] = source.get("color_review_status", "accepted")
                for key, value in source.items():
                    if key.startswith("color_teacher_") or key.startswith("stage52_"):
                        row[key] = value

    candidates = []
    seen_dhash = set()
    for path, row in sorted(merged_supplement.items()):
        dhash = row.get("dhash64", "")
        if len(dhash) != 16:
            rejected["invalid_dhash"] += 1
            continue
        if dhash in base_dhash:
            rejected["base_exact_dhash_overlap"] += 1
            continue
        if dhash in seen_dhash:
            rejected["supplement_exact_dhash_overlap"] += 1
            continue
        seen_dhash.add(dhash)
        heads = source_heads[path]
        row["review_status"] = "approved"
        row["formal_train_eligible"] = "true"
        row["license_train_eligible"] = "true"
        row["pseudo_label"] = "true"
        row["label_confidence"] = "high"
        row["source_group"] = "Open-Images-V7:existing-hard-real"
        row["annotation_source"] = "openimages_hard_multiteacher_consensus"
        row["training_reason"] = "stage59_hard_" + "_and_".join(sorted(heads))
        row["sample_weight"] = "0.500" if len(heads) == 2 else ("0.450" if "body" in heads else "0.350")
        row["small_target"] = str(row.get("vehicle_size") == "small").lower()
        conditions = [
            name for name, present in (
                ("small", row.get("vehicle_size") == "small"),
                ("occluded", truthy(row.get("occluded"))),
                ("truncated", truthy(row.get("truncated"))),
            ) if present
        ]
        row["hard_example_priority"] = "high"
        row["hard_mining_tags"] = ";".join([*conditions, *sorted(heads), "openimages_real_hard"])
        row["hard_score"] = f"{min(8.0, 4.0 + len(conditions)):.4f}"
        row["crop_quality_source"] = row.get("crop_quality", "")
        row["crop_quality"] = "usable"
        is_blurred, blur_score = measure_blur(args.dataset_root / path)
        row["blur"] = str(is_blurred).lower()
        row["blur_metric"] = "variance_of_laplacian_gray"
        row["blur_score"] = f"{blur_score:.6f}"
        row["blur_threshold"] = "60.000000"
        row["review_method"] = row.get("review_method", "") + "+stage59_manifest_merge"
        candidates.append(row)

    fields = list(base_fields)
    for field in [*body_fields, *color_fields]:
        if field not in fields:
            fields.append(field)
    output_rows = [*base_rows, *candidates]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    merged_eval = [row for row in output_rows if row.get("split") in {"validation", "test"}]
    merged_eval_digest = rows_digest(merged_eval, base_fields)
    supplement_counts = Counter()
    for row in candidates:
        supplement_counts["body_supervised"] += truthy(row.get("body_type_supervised"))
        supplement_counts["color_supervised"] += truthy(row.get("color_supervised"))
        supplement_counts["both_supervised"] += truthy(row.get("body_type_supervised")) and truthy(row.get("color_supervised"))
        supplement_counts["small"] += row.get("vehicle_size") == "small"
        supplement_counts["occluded"] += truthy(row.get("occluded"))
        supplement_counts["truncated"] += truthy(row.get("truncated"))
    total = len(candidates)
    quotas = {
        "small_fraction": supplement_counts["small"] / total if total else 0.0,
        "small_fraction_at_least_0_20": supplement_counts["small"] / total >= 0.20 if total else False,
        "occluded_fraction": supplement_counts["occluded"] / total if total else 0.0,
        "occluded_fraction_at_least_0_15": supplement_counts["occluded"] / total >= 0.15 if total else False,
        "truncated_fraction": supplement_counts["truncated"] / total if total else 0.0,
        "truncated_fraction_at_least_0_15": supplement_counts["truncated"] / total >= 0.15 if total else False,
    }
    split_counts = Counter(row.get("split", "") for row in output_rows)
    eval_unchanged = base_eval_digest == merged_eval_digest and len(base_test_validation) == len(merged_eval)
    quota_gates = [value for key, value in quotas.items() if "at_least" in key]
    status = "pass_preweather_only" if (
        total >= 1500 and eval_unchanged and all(quota_gates)
        and not rejected["base_image_path_overlap"] and not rejected["base_group_overlap"]
    ) else "fail"
    report = {
        "schema_version": "attribute-stage59-preweather-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "body_manifest": str(args.body_manifest.resolve()),
        "body_manifest_sha256": sha256(args.body_manifest),
        "body_report": str(args.body_report.resolve()),
        "body_report_sha256": sha256(args.body_report),
        "color_manifest": str(args.color_manifest.resolve()),
        "color_manifest_sha256": sha256(args.color_manifest),
        "color_report": str(args.color_report.resolve()),
        "color_report_sha256": sha256(args.color_report),
        "base_rows": len(base_rows),
        "base_split_counts": dict(sorted(Counter(row.get("split", "") for row in base_rows).items())),
        "supplement_input_rows": {"body": len(body_rows), "color": len(color_rows)},
        "supplement_merged_rows": total,
        "supplement_unique_source_images": len(candidates),
        "supplement_counts": dict(sorted(supplement_counts.items())),
        "supplement_body_counts": dict(sorted(Counter(
            row["body_type"] for row in candidates if truthy(row.get("body_type_supervised"))
        ).items())),
        "supplement_color_counts": dict(sorted(Counter(
            row["color"] for row in candidates if truthy(row.get("color_supervised"))
        ).items())),
        "supplement_quotas": quotas,
        "rejection_counts": dict(sorted(rejected.items())),
        "output_rows": len(output_rows),
        "output_split_counts": dict(sorted(split_counts.items())),
        "evaluation_rows_unchanged": eval_unchanged,
        "base_evaluation_rows_digest": base_eval_digest,
        "output_evaluation_rows_digest": merged_eval_digest,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "input_curation_summaries": {"body": body_report, "color": color_report},
        "policy": {
            "base_rows_preserved": True,
            "supplements_train_only": True,
            "validation_and_test_rows_unchanged": eval_unchanged,
            "exact_path_group_and_dhash_leakage_checked": True,
            "pseudo_labels_low_weight": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "preweather manifest only; wait for licensed weather plan and final merged-distribution audit before training"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "rows": len(output_rows),
        "supplement": total,
        "splits": dict(sorted(split_counts.items())),
        "quotas": quotas,
        "rejections": dict(sorted(rejected.items())),
        "output_sha256": report["output_manifest_sha256"]
    }, ensure_ascii=False))
    return 0 if status == "pass_preweather_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
