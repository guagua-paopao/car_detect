#!/usr/bin/env python3
"""Apply completed review-queue decisions to a new attribute manifest."""

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
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "config" / "vehicle_labels.v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    allowed = {
        "body_type": set(labels["body_types"]),
        "color": set(labels["colors"]),
        "crop_quality": set(labels["crop_qualities"]),
        "viewpoint": set(labels["viewpoints"]),
    }
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        manifest_fields = list(reader.fieldnames)
        rows = list(reader)
    with args.reviews.open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    decisions: dict[str, dict[str, str]] = {}
    for line_number, review in enumerate(review_rows, start=2):
        values = {
            field: review.get(f"reviewed_{field}", "").strip()
            for field in FIELDS
        }
        if not any(values.values()):
            continue
        if not all(values.values()):
            raise ValueError(
                f"{args.reviews}:{line_number}: all four reviewed fields are required"
            )
        for field, value in values.items():
            if value not in allowed[field]:
                raise ValueError(
                    f"{args.reviews}:{line_number}: invalid {field} {value!r}"
                )
        if values["crop_quality"] == "poor" and (
            values["body_type"] != "unknown" or values["color"] != "unknown"
        ):
            raise ValueError(
                f"{args.reviews}:{line_number}: poor requires both attributes unknown"
            )
        decisions[review["image_path"]] = values

    if not decisions:
        raise RuntimeError("review CSV contains no completed decisions")
    for field in (
        "annotation_source",
        "body_type_supervised",
        "color_supervised",
    ):
        if field not in manifest_fields:
            manifest_fields.append(field)
    applied = 0
    for row in rows:
        decision = decisions.get(row["image_path"])
        if decision is None:
            continue
        row.update(decision)
        row["review_status"] = "approved"
        row["annotation_source"] = "human_rereview"
        # A completed human review is authoritative for both attribute heads.
        # Poor crops remain supervised as the explicit unknown/unknown rule.
        row["body_type_supervised"] = "true"
        row["color_supervised"] = "true"
        applied += 1
    missing = set(decisions) - {row["image_path"] for row in rows}
    if missing:
        raise RuntimeError(f"review paths are not present in the manifest: {sorted(missing)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: applied {applied} reviewed decisions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
