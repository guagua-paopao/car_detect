#!/usr/bin/env python3
"""Build a curated manifest by removing bad crops and applying two review sets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import load_json  # noqa: E402


FIELDS = ("body_type", "color", "crop_quality", "viewpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-reviews", type=Path, required=True)
    parser.add_argument("--replacement-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    return parser.parse_args()


def read_reviews(
    path: Path,
    allowed: dict[str, set[str]],
) -> tuple[dict[str, dict[str, str]], set[str]]:
    decisions: dict[str, dict[str, str]] = {}
    rejected: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            values = {
                field: row.get(f"reviewed_{field}", "").strip()
                for field in FIELDS
            }
            if not any(values.values()):
                continue
            if not all(values.values()):
                raise ValueError(f"{path}:{line_number}: incomplete review")
            for field, value in values.items():
                if value not in allowed[field]:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {field} {value!r}"
                    )
            image_path = row["image_path"]
            if values["crop_quality"] == "poor":
                if (
                    values["body_type"] != "unknown"
                    or values["color"] != "unknown"
                ):
                    raise ValueError(
                        f"{path}:{line_number}: poor requires unknown attributes"
                    )
                rejected.add(image_path)
            else:
                decisions[image_path] = values
    return decisions, rejected


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    allowed = {
        "body_type": set(labels["body_types"]),
        "color": set(labels["colors"]),
        "crop_quality": set(labels["crop_qualities"]),
        "viewpoint": set(labels["viewpoints"]),
    }
    original, rejected = read_reviews(args.original_reviews, allowed)
    replacement, replacement_rejected = read_reviews(
        args.replacement_reviews,
        allowed,
    )
    if replacement_rejected:
        raise ValueError("replacement reviews must omit rejected candidates")
    overlap = set(original) & set(replacement)
    if overlap:
        raise ValueError(f"review sets overlap: {sorted(overlap)[:5]}")
    decisions = {**original, **replacement}

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    for field in (
        "annotation_source",
        "body_type_supervised",
        "color_supervised",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    manifest_paths = {row["image_path"] for row in rows}
    missing = (set(decisions) | rejected) - manifest_paths
    if missing:
        raise ValueError(f"review paths missing from manifest: {sorted(missing)[:5]}")

    output_rows = []
    applied = 0
    for row in rows:
        image_path = row["image_path"]
        if image_path in rejected:
            continue
        decision = decisions.get(image_path)
        if decision is not None:
            row.update(decision)
            row["review_status"] = "approved"
            row["annotation_source"] = "human_rereview"
            row["body_type_supervised"] = "true"
            row["color_supervised"] = "true"
            applied += 1
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"PASS: rows={len(output_rows)} removed={len(rejected)} "
        f"applied={applied} original_good={len(original)} "
        f"replacements={len(replacement)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
