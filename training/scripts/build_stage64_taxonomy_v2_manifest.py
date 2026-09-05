#!/usr/bin/env python3
"""Build a fail-closed offline taxonomy-v2 manifest without changing splits.

The current production taxonomy merges gray/silver, yellow/orange, and
brown/beige.  A v2 model must not inherit those merged labels as fabricated
fine-grained truth.  Only VCoR rows whose original folder label is retained
are promoted to fine-grained colors.  Other merged labels are demoted to
unknown and excluded from the corresponding supervised loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


VCOR_SOURCES = {"vcor", "vcor-large", "vcor-stage64-fine"}
FINE_COLORS = {
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
}
VCOR_COLOR_MAP = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "brown": "brown",
    # These are valid colors, but not aliases for a requested fine class.
    "beige": "other",
    "tan": "other",
    "orange": "other",
    "gold": "other",
    "pink": "other",
    "purple": "other",
}
V2_BODY_TYPES = {
    "sedan",
    "suv",
    "mpv",
    "van",
    "pickup",
    "truck",
    "bus",
    "light_truck",
    "heavy_truck",
    "other",
    "unknown",
}
FORBIDDEN_FROZEN_MARKERS = (
    "vcas_rtsp_demo_60s",
    "36-48s",
    "36_48s",
    "frozen_video",
)
STAGE64_METADATA_FIELDS = (
    "source_group",
    "source_manifest",
    "source_license",
    "annotation_source",
    "review_status",
    "review_method",
    "formal_train_eligible",
    "license_train_eligible",
    "sha256",
    "taxonomy_v2_color_source",
    "taxonomy_v2_body_source",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def original_vcor_color(row: dict[str, str]) -> str | None:
    explicit = str(row.get("source_color", "")).strip().lower()
    if explicit:
        return explicit
    value = str(row.get("source_frame_id", "")).strip().lower().replace("\\", "/")
    if value.startswith("vcor:"):
        value = value[len("vcor:") :]
    parts = [part for part in value.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"train", "test", "val", "validation"}:
        return parts[1]
    return None


def migrate_color(row: dict[str, str], counters: Counter[str]) -> None:
    if not is_true(row.get("color_supervised")):
        row["color"] = "unknown"
        row["color_supervised"] = "false"
        return

    source = str(row.get("source_dataset", "")).strip().lower()
    current = str(row.get("color", "")).strip().lower()
    if source in VCOR_SOURCES:
        original = original_vcor_color(row)
        mapped = VCOR_COLOR_MAP.get(str(original or ""))
        if mapped is not None:
            row["color"] = mapped
            row["color_supervised"] = "true"
            row["taxonomy_v2_color_source"] = f"official_vcor_folder:{original}"
            counters[f"vcor_color:{original}->{mapped}"] += 1
            return
        counters["vcor_missing_or_unrecognized_original_color"] += 1

    if current in FINE_COLORS or current == "other":
        row["color"] = current
        row["color_supervised"] = "true"
        row["taxonomy_v2_color_source"] = "unchanged_exact_v1_label"
        counters[f"exact_color_retained:{current}"] += 1
        return

    row["color"] = "unknown"
    row["color_supervised"] = "false"
    row["taxonomy_v2_color_source"] = f"demoted_ambiguous_v1:{current or 'missing'}"
    counters[f"ambiguous_color_demoted:{current or 'missing'}"] += 1


def migrate_body_type(row: dict[str, str], counters: Counter[str]) -> None:
    if not is_true(row.get("body_type_supervised")):
        row["body_type"] = "unknown"
        row["body_type_supervised"] = "false"
        return

    source = str(row.get("source_dataset", "")).strip().lower()
    official = str(row.get("official_vehicle_class", "")).strip().lower()
    current = str(row.get("body_type", "")).strip().lower()
    if source == "open-images-v7" and official == "truck":
        row["body_type"] = "truck"
        row["body_type_supervised"] = "true"
        row["taxonomy_v2_body_source"] = "official_openimages_coarse_truck"
        counters[f"openimages_truck_remapped:{current}->truck"] += 1
        return

    if current in V2_BODY_TYPES:
        row["body_type"] = current
        row["body_type_supervised"] = "true"
        row["taxonomy_v2_body_source"] = "unchanged_v1_label"
        counters[f"body_retained:{current}"] += 1
        return

    row["body_type"] = "unknown"
    row["body_type_supervised"] = "false"
    row["taxonomy_v2_body_source"] = f"demoted_unrecognized_v1:{current or 'missing'}"
    counters[f"unrecognized_body_demoted:{current or 'missing'}"] += 1


def frozen_marker(row: dict[str, str]) -> str | None:
    searchable = " ".join(str(value).lower() for value in row.values())
    return next((marker for marker in FORBIDDEN_FROZEN_MARKERS if marker in searchable), None)


def count_labels(rows: list[dict[str, str]], head: str) -> dict[str, int]:
    label = "body_type" if head == "body" else "color"
    supervised = "body_type_supervised" if head == "body" else "color_supervised"
    counts = Counter(
        f"{row.get('split', 'unknown')}:{row.get(label, 'unknown')}"
        for row in rows
        if is_true(row.get(supervised))
    )
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--vcor-source-manifest",
        type=Path,
        help="Optional original VCoR manifest; only unseen official train rows are appended.",
    )
    return parser.parse_args()


def append_vcor_train_rows(
    rows: list[dict[str, str]],
    input_fields: list[str],
    source_manifest: Path | None,
    output_manifest: Path,
) -> tuple[int, Counter[str]]:
    skipped: Counter[str] = Counter()
    if source_manifest is None:
        return 0, skipped

    existing_sha = {row.get("sha256", "") for row in rows if row.get("sha256")}
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    added = 0
    for source_row in source_rows:
        if str(source_row.get("split", "")).strip().lower() != "train":
            skipped["heldout_source_row_not_appended"] += 1
            continue
        if not is_true(source_row.get("color_supervised")):
            skipped["not_color_supervised"] += 1
            continue
        digest = str(source_row.get("sha256", "")).strip()
        if not digest:
            skipped["missing_sha256"] += 1
            continue
        if digest in existing_sha:
            skipped["duplicate_sha256"] += 1
            continue
        source_image = (source_manifest.parent / source_row.get("image_path", "")).resolve()
        if not source_image.is_file():
            skipped["missing_image"] += 1
            continue

        row = {field: source_row.get(field, "") for field in input_fields}
        row["image_path"] = os.path.relpath(
            source_image, output_manifest.parent.resolve()
        ).replace("\\", "/")
        row["split"] = "train"
        row["source_dataset"] = "VCoR-STAGE64-FINE"
        row["source_group"] = "VCoR-STAGE64-FINE"
        row["source_manifest"] = str(source_manifest)
        original_id = str(source_row.get("source_frame_id", "")).strip()
        row["source_frame_id"] = (
            original_id if original_id.lower().startswith("vcor:") else f"vcor:{original_id}"
        )
        row["source_license"] = source_row.get("source_license", "")
        row["annotation_source"] = "official_dataset"
        row["review_status"] = "approved"
        row["review_method"] = "official_vcor_folder+existing_quality_audit+sha_dedup"
        row["formal_train_eligible"] = "true"
        row["license_train_eligible"] = "true"
        row["body_type"] = "unknown"
        row["body_type_supervised"] = "false"
        row["color_supervised"] = "true"
        row["sha256"] = digest
        rows.append(row)
        existing_sha.add(digest)
        added += 1
    return added, skipped


def main() -> None:
    args = parse_args()
    if args.input_manifest.resolve() == args.output_manifest.resolve():
        raise RuntimeError("output manifest must not overwrite the input manifest")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("the Stage64 builder requires the offline taxonomy-v2 labels file")

    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("input manifest has no header")
        input_fields = list(reader.fieldnames)
        rows = list(reader)

    base_rows = len(rows)
    base_split_counts = Counter(row.get("split", "unknown") for row in rows)
    appended_vcor_train_rows, append_skipped = append_vcor_train_rows(
        rows,
        input_fields,
        args.vcor_source_manifest,
        args.output_manifest,
    )

    for index, row in enumerate(rows, start=2):
        marker = frozen_marker(row)
        if marker:
            raise RuntimeError(f"frozen-video isolation violation at CSV line {index}: {marker}")

    fields = input_fields + [
        field for field in STAGE64_METADATA_FIELDS if field not in input_fields
    ]
    counters: Counter[str] = Counter()
    for row in rows:
        migrate_color(row, counters)
        migrate_body_type(row, counters)
    split_after = Counter(row.get("split", "unknown") for row in rows)
    if split_after.get("validation", 0) != base_split_counts.get("validation", 0):
        raise RuntimeError("taxonomy migration changed validation membership")
    if split_after.get("test", 0) != base_split_counts.get("test", 0):
        raise RuntimeError("taxonomy migration changed test membership")
    expected_train = base_split_counts.get("train", 0) + appended_vcor_train_rows
    if split_after.get("train", 0) != expected_train:
        raise RuntimeError("unexpected train row count after VCoR append")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": "stage64-taxonomy-v2-manifest-report-v1",
        "status": "pass",
        "deployment_status": "offline_candidate_only",
        "input_manifest": str(args.input_manifest),
        "input_manifest_sha256": sha256(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
        "labels": str(args.labels),
        "labels_sha256": sha256(args.labels),
        "base_rows": base_rows,
        "appended_vcor_train_rows": appended_vcor_train_rows,
        "append_skipped": dict(sorted(append_skipped.items())),
        "rows": len(rows),
        "split_counts": dict(sorted(split_after.items())),
        "body_supervised_counts": count_labels(rows, "body"),
        "color_supervised_counts": count_labels(rows, "color"),
        "migration_counters": dict(sorted(counters.items())),
        "policy": {
            "existing_split_assignments_preserved_exactly": True,
            "validation_and_test_membership_preserved_exactly": True,
            "only_unseen_official_vcor_train_rows_may_be_appended": True,
            "no_source_rows_removed": True,
            "merged_v1_colors_demoted_without_fine_truth": True,
            "vcor_original_folder_truth_required_for_fine_color_migration": True,
            "generic_truck_requires_official_coarse_truck_truth": True,
            "frozen_video_used": False,
            "production_contract_modified": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
