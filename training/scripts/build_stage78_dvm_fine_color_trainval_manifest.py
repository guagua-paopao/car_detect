#!/usr/bin/env python3
"""Build a leakage-clean DVM taxonomy-v2 fine-color train/validation manifest.

Only retained official DVM ``source_color`` metadata is used. The existing
group-held-out validation split remains validation and is never moved into
training. No pixel heuristic, v1 teacher, test row, or frozen-video asset is
accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "frozen_video")
FINE_MAP = {
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
V1_MAP = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def transform_row(
    source: dict[str, str], parent_sha256: str
) -> tuple[dict[str, str] | None, str | None]:
    text = " ".join(str(value).lower() for value in source.values())
    marker = next((item for item in FROZEN_MARKERS if item in text), None)
    if marker:
        return None, f"frozen marker {marker}"
    split = str(source.get("split", "")).strip().lower()
    if split == "test":
        return None, "test row present"
    if split not in {"train", "validation"}:
        return None, f"unsupported split {split!r}"
    dataset = str(source.get("source_dataset", "")).strip().lower()
    if dataset not in {"dvm-car-2.0", "dvm-car 2.0"}:
        return None, f"unexpected source dataset {dataset!r}"
    license_text = str(source.get("source_license", "")).strip().lower()
    if "cc-by-nc-4.0" not in license_text and "cc by-nc 4.0" not in license_text:
        return None, "DVM CC BY-NC evidence missing"
    if not truthy(source.get("color_supervised")):
        return None, "source color is not supervised"
    source_color = str(source.get("source_color", "")).strip()
    fine = FINE_MAP.get(source_color)
    expected_v1 = V1_MAP.get(source_color)
    if fine is None or expected_v1 is None:
        return None, f"unsupported official source_color {source_color!r}"
    if str(source.get("color", "")).strip().lower() != expected_v1:
        return None, "official source_color conflicts with audited v1 label"
    group = str(source.get("source_group", "")).strip()
    digest = str(source.get("sha256", "")).strip().lower()
    if not group:
        return None, "missing source_group"
    if not HEX64.fullmatch(digest):
        return None, "invalid sha256"
    row = dict(source)
    row["split"] = split
    row["body_type"] = "unknown"
    row["body_type_supervised"] = "false"
    row["color"] = fine
    row["color_supervised"] = "true"
    row["taxonomy_v2_color_source"] = f"official_dvm_folder:{source_color}"
    row["taxonomy_v2_eligibility"] = "research-only_non-deployable"
    row["taxonomy_v2_parent_manifest_sha256"] = parent_sha256
    return row, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-train-rows", type=int, default=118338)
    parser.add_argument("--minimum-validation-rows", type=int, default=6598)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage78 evidence")
    parent_sha = sha256(args.parent_manifest)
    if parent_sha.lower() != args.expected_parent_sha256.lower():
        raise RuntimeError("parent manifest SHA256 mismatch")
    with args.parent_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_FIELDS - set(fields))
        if missing:
            raise RuntimeError(f"missing required fields: {missing}")
        source_rows = list(reader)

    failures: list[str] = []
    output_rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    digest_splits: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for line, source in enumerate(source_rows, start=2):
        row, error = transform_row(source, parent_sha)
        if error:
            failures.append(f"line {line}: {error}")
            continue
        assert row is not None
        digest = row["sha256"].lower()
        if digest in digest_splits:
            failures.append(f"line {line}: duplicate sha256")
            continue
        digest_splits[digest] = row["split"]
        group_splits[row["source_group"]].add(row["split"])
        counters[f"{row['split']}:{row['color']}"] += 1
        counters[f"rows:{row['split']}"] += 1
        output_rows.append(row)

    group_leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if group_leaks:
        failures.append(f"source_group cross-split leaks: {len(group_leaks)}")
    requirements = {
        "train_rows": (counters["rows:train"], args.minimum_train_rows),
        "validation_rows": (
            counters["rows:validation"],
            args.minimum_validation_rows,
        ),
        "validation_gray": (counters["validation:gray"], 400),
        "validation_silver": (counters["validation:silver"], 400),
        "validation_yellow": (counters["validation:yellow"], 150),
        "validation_brown": (counters["validation:brown"], 150),
    }
    for name, (actual, minimum) in requirements.items():
        if actual < minimum:
            failures.append(f"{name}: {actual} < {minimum}")

    report = {
        "schema_version": "stage78-dvm-fine-color-trainval-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "input": {
            "manifest": str(args.parent_manifest.resolve()),
            "sha256": parent_sha,
            "rows": len(source_rows),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "rows": len(output_rows),
            "counts": dict(sorted(counters.items())),
        },
        "requirements": {
            key: {"actual": actual, "minimum": minimum, "pass": actual >= minimum}
            for key, (actual, minimum) in requirements.items()
        },
        "integrity": {
            "source_group_cross_split_leaks": len(group_leaks),
            "duplicate_sha256": sum("duplicate sha256" in item for item in failures),
            "test_rows": sum("test row" in item for item in failures),
            "frozen_markers": sum("frozen marker" in item for item in failures),
        },
        "policy": {
            "mapping_source": "official DVM source_color metadata only",
            "validation_split_preserved": True,
            "pixel_heuristic_used": False,
            "merged_v1_label_used_to_split_classes": False,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "CC BY-NC research-only_non-deployable",
        },
        "failures": failures[:200],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    atomic = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    atomic.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(atomic, args.output_report)
    if failures:
        print(json.dumps({"status": "fail", "failures": len(failures)}))
        return 2

    output_fields = fields + [field for field in ADDED_FIELDS if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    report["output"]["sha256"] = sha256(args.output_manifest)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "rows": len(output_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
