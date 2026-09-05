#!/usr/bin/env python3
"""Recover taxonomy-v2 color truth from audited DVM-CAR source metadata.

Stage68 intentionally trained the production-compatible v1 taxonomy, which
merged Grey/Silver, Yellow/Orange/Gold, and Brown/Beige/Bronze.  Its manifest
nevertheless retained the official DVM ``source_color`` folder value.  This
builder uses only that retained official metadata to create a train-only v2
color manifest.  It never infers a fine color from pixels or from a merged v1
label.

The resulting data remains CC BY-NC research-only and non-deployable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = (
    "vcas_rtsp_demo_60s",
    "36-48s",
    "36_48s",
    "frozen_video",
)

# Exact requested classes are recovered only from exact official folder names.
# Related but distinct paint names are conservatively routed to ``other``.
DVM_FINE_COLOR_MAP = {
    "Black": "black",
    "White": "white",
    "Silver": "silver",
    "Grey": "gray",
    "Red": "red",
    "Blue": "blue",
    "Navy": "blue",
    "Indigo": "blue",
    "Green": "green",
    "Yellow": "yellow",
    "Brown": "brown",
    "Orange": "other",
    "Gold": "other",
    "Beige": "other",
    "Bronze": "other",
    "Purple": "other",
    "Pink": "other",
    "Magenta": "other",
    "Turquoise": "other",
    "Multicolour": "other",
}

# This is evidence checking, not the v2 mapping.  It proves the retained source
# metadata agrees with the v1 label produced by the audited Stage68 builder.
DVM_V1_COLOR_MAP = {
    "Black": "black",
    "White": "white",
    "Silver": "silver_gray",
    "Grey": "silver_gray",
    "Red": "red",
    "Blue": "blue",
    "Navy": "blue",
    "Indigo": "blue",
    "Green": "green",
    "Yellow": "yellow_orange",
    "Orange": "yellow_orange",
    "Gold": "yellow_orange",
    "Brown": "brown_beige",
    "Beige": "brown_beige",
    "Bronze": "brown_beige",
    "Purple": "other",
    "Pink": "other",
    "Magenta": "other",
    "Turquoise": "other",
    "Multicolour": "other",
}

REQUIRED_FIELDS = {
    "image_path",
    "split",
    "color",
    "color_supervised",
    "source_dataset",
    "source_color",
    "source_license",
    "source_group",
    "sha256",
}
ADDED_FIELDS = (
    "taxonomy_v2_color_source",
    "taxonomy_v2_eligibility",
    "taxonomy_v2_parent_manifest_sha256",
)
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def searchable(row: dict[str, str]) -> str:
    return " ".join(str(value).lower() for value in row.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dvm-manifest", type=Path, required=True)
    parser.add_argument("--expected-dvm-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-train-rows", type=int, default=118338)
    parser.add_argument("--minimum-gray-rows", type=int, default=1000)
    parser.add_argument("--minimum-silver-rows", type=int, default=1000)
    parser.add_argument("--minimum-yellow-rows", type=int, default=500)
    parser.add_argument("--minimum-brown-rows", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage72 evidence")
    if not args.dvm_manifest.is_file():
        raise FileNotFoundError(args.dvm_manifest)

    parent_sha = file_sha256(args.dvm_manifest)
    if parent_sha.lower() != args.expected_dvm_sha256.lower():
        raise RuntimeError("DVM manifest SHA256 mismatch")

    with args.dvm_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_FIELDS - set(fields))
        if missing:
            raise RuntimeError(f"DVM manifest missing required fields: {missing}")
        source_rows = list(reader)

    counters: Counter[str] = Counter()
    failures: list[str] = []
    output_rows: list[dict[str, str]] = []
    sha_to_label: dict[str, str] = {}
    group_labels: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)

    for index, source in enumerate(source_rows, start=2):
        row_text = searchable(source)
        marker = next((item for item in FROZEN_MARKERS if item in row_text), None)
        if marker:
            failures.append(f"line {index}: frozen marker {marker}")
            continue

        split = str(source.get("split", "")).strip().lower()
        if split == "test":
            failures.append(f"line {index}: test row present in DVM source manifest")
            continue
        if split not in {"train", "validation"}:
            failures.append(f"line {index}: unsupported split {split!r}")
            continue

        group = str(source.get("source_group", "")).strip()
        if not group:
            failures.append(f"line {index}: missing source_group")
            continue
        split_groups[split].add(group)

        # Validation stays held out and is never copied into this training pool.
        if split != "train":
            counters["heldout_validation_rows_excluded"] += 1
            continue

        dataset = str(source.get("source_dataset", "")).strip().lower()
        license_text = str(source.get("source_license", "")).strip().lower()
        if dataset not in {"dvm-car-2.0", "dvm-car 2.0"}:
            failures.append(f"line {index}: unexpected source_dataset {dataset!r}")
            continue
        if "cc-by-nc-4.0" not in license_text and "cc by-nc 4.0" not in license_text:
            failures.append(f"line {index}: DVM CC BY-NC license evidence missing")
            continue
        if not true(source.get("color_supervised")):
            failures.append(f"line {index}: train row is not color supervised")
            continue

        source_color = str(source.get("source_color", "")).strip()
        fine = DVM_FINE_COLOR_MAP.get(source_color)
        expected_v1 = DVM_V1_COLOR_MAP.get(source_color)
        if fine is None or expected_v1 is None:
            failures.append(f"line {index}: unrecognized official source_color {source_color!r}")
            continue
        current = str(source.get("color", "")).strip().lower()
        if current != expected_v1:
            failures.append(
                f"line {index}: source_color {source_color!r} conflicts with v1 color {current!r}"
            )
            continue

        digest = str(source.get("sha256", "")).strip().lower()
        if not HEX64.fullmatch(digest):
            failures.append(f"line {index}: invalid or missing sha256")
            continue
        previous = sha_to_label.get(digest)
        if previous is not None:
            failures.append(
                f"line {index}: duplicate sha256 in source manifest ({previous} vs {fine})"
            )
            continue
        sha_to_label[digest] = fine
        group_labels[group].add(fine)

        row = dict(source)
        row["split"] = "train"
        row["color"] = fine
        row["color_supervised"] = "true"
        row["body_type"] = "unknown"
        row["body_type_supervised"] = "false"
        row["taxonomy_v2_color_source"] = f"official_dvm_folder:{source_color}"
        row["taxonomy_v2_eligibility"] = "research-only_non-deployable"
        row["taxonomy_v2_parent_manifest_sha256"] = parent_sha
        output_rows.append(row)
        counters[f"train_color:{fine}"] += 1
        counters[f"official_source_color:{source_color}"] += 1

    leaking_groups = sorted(split_groups["train"] & split_groups["validation"])
    if leaking_groups:
        failures.append(f"train/validation source_group leakage: {len(leaking_groups)} groups")
    conflicting_groups = sorted(group for group, labels in group_labels.items() if len(labels) > 1)
    if conflicting_groups:
        failures.append(f"conflicting fine colors inside {len(conflicting_groups)} train groups")

    requirements = {
        "minimum_train_rows": (len(output_rows), args.minimum_train_rows),
        "minimum_gray_rows": (counters["train_color:gray"], args.minimum_gray_rows),
        "minimum_silver_rows": (counters["train_color:silver"], args.minimum_silver_rows),
        "minimum_yellow_rows": (counters["train_color:yellow"], args.minimum_yellow_rows),
        "minimum_brown_rows": (counters["train_color:brown"], args.minimum_brown_rows),
    }
    for name, (actual, minimum) in requirements.items():
        if actual < minimum:
            failures.append(f"{name}: {actual} < {minimum}")

    status = "pass" if not failures else "fail"
    report = {
        "schema_version": "attribute-stage72-dvm-fine-color-manifest-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest": str(args.dvm_manifest.resolve()),
            "sha256": parent_sha,
            "rows": len(source_rows),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "rows": len(output_rows),
            "train_colors": dict(
                sorted(
                    (key.split(":", 1)[1], value)
                    for key, value in counters.items()
                    if key.startswith("train_color:")
                )
            ),
        },
        "requirements": {
            name: {"actual": actual, "minimum": minimum, "pass": actual >= minimum}
            for name, (actual, minimum) in requirements.items()
        },
        "counters": dict(sorted(counters.items())),
        "integrity": {
            "train_validation_group_leaks": len(leaking_groups),
            "conflicting_train_groups": len(conflicting_groups),
            "duplicate_sha256": sum("duplicate sha256" in item for item in failures),
            "frozen_markers": sum("frozen marker" in item for item in failures),
            "test_rows": sum("test row" in item for item in failures),
        },
        "policy": {
            "mapping_source": "official DVM source_color metadata only",
            "pixel_heuristic_used": False,
            "merged_v1_label_used_to_split_classes": False,
            "validation_rows_imported": False,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "CC BY-NC research-only_non-deployable",
        },
        "failures": failures[:200],
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps({"status": status, "failures": len(failures)}, ensure_ascii=False))
        return 2

    output_fields = fields + [field for field in ADDED_FIELDS if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    report["output"]["sha256"] = file_sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(output_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
