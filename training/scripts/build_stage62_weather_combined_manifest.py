#!/usr/bin/env python3
"""Merge audited hard+weather attributes into a locked base with near-dedup gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
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


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def rows_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {field: row.get(field, "") for field in fields}
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_curation(report_path: Path, manifest_path: Path, head: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"{head} combined curation did not pass")
    if report.get("output_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(f"{head} combined curation manifest hash mismatch")
    policy = report.get("policy", {})
    if not (
        policy.get("train_only") is True
        and policy.get("validation_or_test_used") is False
        and policy.get("frozen_video_used") is False
        and policy.get("production_model_modified") is False
        and policy.get("deployment_performed") is False
    ):
        raise RuntimeError(f"{head} combined curation policy is not isolated")
    return report


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def bucket_keys(value: str) -> list[tuple[int, str]]:
    return [(index, value[index * 4:(index + 1) * 4]) for index in range(4)]


def add_hash(buckets: dict[tuple[int, str], set[str]], value: str) -> None:
    for key in bucket_keys(value):
        buckets[key].add(value)


def near_duplicate(
    buckets: dict[tuple[int, str], set[str]], value: str, maximum_distance: int
) -> bool:
    possible: set[str] = set()
    for key in bucket_keys(value):
        possible.update(buckets.get(key, set()))
    return any(hamming(value, other) <= maximum_distance for other in possible)


def measure_blur(path: Path, threshold: float) -> tuple[bool, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise RuntimeError(f"could not read supplement crop: {path}")
    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return score < threshold, score


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
    parser.add_argument("--max-dhash-distance", type=int, default=3)
    parser.add_argument("--blur-threshold", type=float, default=60.0)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage62 manifest evidence")

    body_report = validate_curation(args.body_report, args.body_manifest, "body")
    color_report = validate_curation(args.color_report, args.color_manifest, "color")
    base_fields, base_rows = load_csv(args.base_manifest)
    body_fields, body_rows = load_csv(args.body_manifest)
    color_fields, color_rows = load_csv(args.color_manifest)
    if any(row.get("split") != "train" for row in body_rows + color_rows):
        raise RuntimeError("combined supplement contains non-train rows")

    dataset_root = args.dataset_root.resolve()
    base_paths = {row.get("image_path", "") for row in base_rows}
    base_groups = {
        value
        for row in base_rows
        for value in (row.get("camera_id", ""), row.get("video_id", ""), row.get("track_group", ""))
        if value
    }
    base_hash_buckets: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in base_rows:
        value = row.get("dhash64", "")
        if len(value) == 16:
            add_hash(base_hash_buckets, value)
    base_eval = [row for row in base_rows if row.get("split") in {"validation", "test"}]
    base_eval_digest = rows_digest(base_eval, base_fields)

    merged: dict[str, dict[str, str]] = {}
    source_heads: dict[str, set[str]] = defaultdict(set)
    source_pools: dict[str, set[str]] = defaultdict(set)
    rejected = Counter()
    for head, rows in (("body", body_rows), ("color", color_rows)):
        for source in rows:
            path = source.get("image_path", "")
            if not path or path in base_paths:
                rejected["base_or_empty_image_path_overlap"] += 1
                continue
            absolute = (dataset_root / path).resolve()
            try:
                absolute.relative_to(dataset_root)
            except ValueError:
                rejected["image_path_outside_dataset_root"] += 1
                continue
            if not absolute.is_file():
                rejected["missing_image"] += 1
                continue
            groups = {
                source.get("camera_id", ""), source.get("video_id", ""), source.get("track_group", "")
            } - {""}
            if groups & base_groups:
                rejected["base_group_overlap"] += 1
                continue
            if path not in merged:
                row = dict(source)
                row["body_type"] = "unknown"
                row["color"] = "unknown"
                row["body_type_supervised"] = "false"
                row["color_supervised"] = "false"
                merged[path] = row
            row = merged[path]
            source_heads[path].add(head)
            source_pools[path].add(source.get("combined_source_pool", "unknown"))
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

    selected_hash_buckets: dict[tuple[int, str], set[str]] = defaultdict(set)
    candidates = []
    for path, row in sorted(merged.items()):
        dhash = row.get("dhash64", "")
        if len(dhash) != 16:
            rejected["invalid_dhash"] += 1
            continue
        if near_duplicate(base_hash_buckets, dhash, args.max_dhash_distance):
            rejected["base_perceptual_near_duplicate"] += 1
            continue
        if near_duplicate(selected_hash_buckets, dhash, args.max_dhash_distance):
            rejected["supplement_perceptual_near_duplicate"] += 1
            continue
        add_hash(selected_hash_buckets, dhash)
        heads = source_heads[path]
        pools = source_pools[path]
        row["review_status"] = "approved"
        row["formal_train_eligible"] = "true"
        row["license_train_eligible"] = "true"
        row["pseudo_label"] = "true"
        row["label_confidence"] = "high"
        row["source_group"] = "Open-Images-V7:combined-hard-weather-real"
        row["annotation_source"] = "openimages_combined_hard_weather_multiteacher_consensus"
        row["training_reason"] = "stage62_" + "_and_".join(sorted(heads))
        row["combined_source_pools"] = ";".join(sorted(pools))
        if len(heads) == 2:
            row["sample_weight"] = "0.600" if "weather" in pools else "0.500"
        elif "body" in heads:
            row["sample_weight"] = "0.550" if "weather" in pools else "0.450"
        else:
            row["sample_weight"] = "0.500" if "weather" in pools else "0.350"
        row["small_target"] = str(row.get("vehicle_size") == "small").lower()
        weather_labels = set(row.get("weather", "").split(";")) - {""}
        conditions = [
            name for name, present in (
                ("night", truthy(row.get("night"))),
                ("rain", "rain" in weather_labels),
                ("fog", "fog" in weather_labels),
                ("snow", "snow" in weather_labels),
                ("small", row.get("vehicle_size") == "small"),
                ("occluded", truthy(row.get("occluded"))),
                ("truncated", truthy(row.get("truncated"))),
            ) if present
        ]
        row["hard_example_priority"] = "high"
        row["hard_mining_tags"] = ";".join([*conditions, *sorted(heads), "openimages_real"])
        row["hard_score"] = f"{min(10.0, 4.0 + len(conditions)):.4f}"
        row["crop_quality_source"] = row.get("crop_quality", "")
        row["crop_quality"] = "usable"
        is_blurred, blur_score = measure_blur(dataset_root / path, args.blur_threshold)
        row["blur"] = str(is_blurred).lower()
        row["blur_metric"] = "variance_of_laplacian_gray"
        row["blur_score"] = f"{blur_score:.6f}"
        row["blur_threshold"] = f"{args.blur_threshold:.6f}"
        row["review_method"] = row.get("review_method", "") + "+stage62_locked_merge"
        candidates.append(row)

    fields = list(base_fields)
    additions = [
        "combined_source_pools", "sample_weight", "source_group", "training_reason",
        "small_target", "hard_example_priority", "hard_mining_tags", "hard_score",
        "crop_quality_source", "blur_metric", "blur_score", "blur_threshold",
    ]
    for field in [*body_fields, *color_fields, *additions]:
        if field not in fields:
            fields.append(field)
    output_rows = [*base_rows, *candidates]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    output_eval = [row for row in output_rows if row.get("split") in {"validation", "test"}]
    output_eval_digest = rows_digest(output_eval, base_fields)
    eval_unchanged = len(base_eval) == len(output_eval) and base_eval_digest == output_eval_digest
    body_counts = Counter(
        row["body_type"] for row in candidates if truthy(row.get("body_type_supervised"))
    )
    color_counts = Counter(
        row["color"] for row in candidates if truthy(row.get("color_supervised"))
    )
    body_supervised = sum(body_counts.values())
    color_supervised = sum(color_counts.values())
    total = len(candidates)
    night = sum(truthy(row.get("night")) for row in candidates)
    small = sum(row.get("vehicle_size") == "small" for row in candidates)
    hard_union = sum(
        row.get("vehicle_size") == "small"
        or truthy(row.get("occluded"))
        or truthy(row.get("truncated"))
        for row in candidates
    )
    neutral = color_counts["white"] + color_counts["silver_gray"]
    minimum_color_count = min(color_counts.values()) if color_counts else 0
    quotas = {
        "night_fraction": night / total if total else 0.0,
        "night_fraction_at_least_0_30": night / total >= 0.30 if total else False,
        "small_fraction": small / total if total else 0.0,
        "small_fraction_at_least_0_20": small / total >= 0.20 if total else False,
        "hard_union_fraction": hard_union / total if total else 0.0,
        "hard_union_fraction_at_least_0_15": hard_union / total >= 0.15 if total else False,
        "white_plus_silver_gray_fraction_of_color": neutral / color_supervised if color_supervised else 0.0,
        "neutral_color_fraction_at_most_0_45": neutral / color_supervised <= 0.45 if color_supervised else False,
    }
    gates = {
        "minimum_1800_unique_supplement_images": total >= 1800,
        "minimum_1500_body_supervised": body_supervised >= 1500,
        "minimum_5_body_classes": len(body_counts) >= 5,
        "minimum_800_color_supervised": color_supervised >= 800,
        "minimum_8_color_classes": len(color_counts) >= 8,
        "minimum_20_per_color": minimum_color_count >= 20,
        "night_quota": quotas["night_fraction_at_least_0_30"],
        "small_quota": quotas["small_fraction_at_least_0_20"],
        "hard_union_quota": quotas["hard_union_fraction_at_least_0_15"],
        "neutral_color_cap": quotas["neutral_color_fraction_at_most_0_45"],
        "evaluation_rows_unchanged": eval_unchanged,
        "no_base_path_or_group_overlap": not (
            rejected["base_or_empty_image_path_overlap"] or rejected["base_group_overlap"]
        ),
    }
    status = "pass" if all(gates.values()) else "fail"
    split_counts = Counter(row.get("split", "") for row in output_rows)
    report = {
        "schema_version": "attribute-stage62-weather-combined-manifest-v1",
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
        "supplement_input_rows": {"body": len(body_rows), "color": len(color_rows)},
        "supplement_unique_images": total,
        "supplement_body_supervised": body_supervised,
        "supplement_color_supervised": color_supervised,
        "supplement_body_counts": dict(sorted(body_counts.items())),
        "supplement_color_counts": dict(sorted(color_counts.items())),
        "supplement_night_rows": night,
        "supplement_small_rows": small,
        "supplement_hard_union_rows": hard_union,
        "supplement_quotas": quotas,
        "rejection_counts": dict(sorted(rejected.items())),
        "max_dhash_distance": args.max_dhash_distance,
        "gates": gates,
        "output_rows": len(output_rows),
        "output_split_counts": dict(sorted(split_counts.items())),
        "base_evaluation_rows_digest": base_eval_digest,
        "output_evaluation_rows_digest": output_eval_digest,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "input_curation_summaries": {"body": body_report, "color": color_report},
        "policy": {
            "base_rows_preserved_in_order_and_fields": True,
            "supplements_train_only": True,
            "validation_and_test_rows_unchanged": eval_unchanged,
            "base_and_supplement_perceptual_near_dedup_checked": True,
            "pseudo_labels_low_weight": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only pass status authorizes validation-only Stage62 candidate screening; test and frozen video remain inaccessible",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "rows": len(output_rows),
        "supplement": total,
        "body_supervised": body_supervised,
        "color_supervised": color_supervised,
        "quotas": quotas,
        "gates": gates,
        "rejections": dict(sorted(rejected.items())),
        "output_sha256": report["output_manifest_sha256"],
    }, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
