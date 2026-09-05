#!/usr/bin/env python3
"""Audit only public Tesla lighting/color metadata before the 8.84 GB image download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED = {
    "tesla_dataset_labels.csv": (194654, "b1dd6b1326612a3ceefdb144f0a9b0d0"),
    "labels_description.txt": (1275, "4f91d949c6c458ebc1f829e01a9f71b0"),
    "label-studio-code.xml": (2149, "d8d0b4a12e99693f736a580c0c0028ba"),
}
EXACT_COLOR = {
    "Blue": "blue",
    "Green": "green",
    "Red": "red",
    "White": "white",
    "Yellow": "yellow",
}
AMBIGUOUS_COLOR = {"Black/Dark gray", "Light gray/Silver", "Orange"}
EXACT_BODY = {"3": "sedan", "S": "sedan", "X": "suv", "Y": "suv"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(root: Path, report_path: Path) -> dict[str, object]:
    for name, (expected_size, expected_md5) in EXPECTED.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected_size or md5(path) != expected_md5:
            raise ValueError(f"metadata integrity failure: {name}")
    labels_path = root / "tesla_dataset_labels.csv"
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"color", "image", "lighting", "model", "year_3", "year_s", "year_x", "year_y"}
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected label columns")
    images = [row["image"] for row in rows]
    colors = Counter(row["color"] for row in rows)
    lighting = Counter(row["lighting"] for row in rows)
    models = Counter(row["model"] for row in rows)
    if len(rows) != 3026 or len(set(images)) != 3026:
        raise ValueError("expected 3026 unique image labels")
    if set(lighting) != {"Light", "Medium", "Dark"}:
        raise ValueError(f"unexpected lighting labels: {sorted(lighting)}")
    if set(colors) != set(EXACT_COLOR) | AMBIGUOUS_COLOR:
        raise ValueError(f"unexpected color labels: {sorted(colors)}")
    if set(models) != set(EXACT_BODY) | {"Other car"}:
        raise ValueError(f"unexpected model labels: {sorted(models)}")

    exact_color = Counter()
    exact_color_by_lighting: dict[str, Counter[str]] = {
        key: Counter() for key in ("Light", "Medium", "Dark")
    }
    exact_body = Counter()
    exact_body_by_lighting: dict[str, Counter[str]] = {
        key: Counter() for key in ("Light", "Medium", "Dark")
    }
    for row in rows:
        color = EXACT_COLOR.get(row["color"])
        body = EXACT_BODY.get(row["model"])
        if color:
            exact_color[color] += 1
            exact_color_by_lighting[row["lighting"]][color] += 1
        if body:
            exact_body[body] += 1
            exact_body_by_lighting[row["lighting"]][body] += 1

    report: dict[str, object] = {
        "stage": "stage189_tesla_lighting_color_metadata_audit_r1",
        "status": "pass_metadata_only_image_archive_pending_no_training",
        "source": {
            "title": "Labeled Tesla Vehicle Image Dataset under Varying Lighting Conditions",
            "doi": "10.5281/zenodo.19814157",
            "license": "CC BY 4.0",
            "publication_date": "2026-04-27",
            "archive": "Tesla-dataset.zip",
            "archive_bytes": 8841072987,
            "archive_md5": "9ce3079d42273f07374bd25344809d2f",
            "archive_url": "https://zenodo.org/api/records/19814157/files/Tesla-dataset.zip/content",
        },
        "metadata": {
            name: {"bytes": (root / name).stat().st_size, "md5": md5(root / name), "sha256": sha256(root / name)}
            for name in EXPECTED
        },
        "counts": {
            "rows": len(rows),
            "unique_image_filenames": len(set(images)),
            "lighting": dict(sorted(lighting.items())),
            "colors": dict(sorted(colors.items())),
            "models": dict(sorted(models.items())),
            "exact_color_rows": sum(exact_color.values()),
            "exact_color": dict(sorted(exact_color.items())),
            "exact_color_by_lighting": {
                key: dict(sorted(value.items())) for key, value in exact_color_by_lighting.items()
            },
            "exact_color_medium_or_dark_rows": sum(
                exact_color_by_lighting[key].total() for key in ("Medium", "Dark")
            ),
            "exact_body_rows": sum(exact_body.values()),
            "exact_body": dict(sorted(exact_body.items())),
            "exact_body_by_lighting": {
                key: dict(sorted(value.items())) for key, value in exact_body_by_lighting.items()
            },
            "exact_body_medium_or_dark_rows": sum(
                exact_body_by_lighting[key].total() for key in ("Medium", "Dark")
            ),
            "confirmed_natural_night_rows": 0,
        },
        "mapping_policy": {
            "Black/Dark gray": "unknown until black versus gray can be independently separated",
            "Light gray/Silver": "unknown until gray versus silver can be independently separated",
            "Orange": "other; not merged into yellow",
            "Dark": "source-explicit low-visibility truth; not guaranteed natural night",
            "Medium": "source-explicit reduced/uneven-light truth; not guaranteed natural night",
            "Light": "daylight/normal visibility",
            "Tesla_3_or_S": "sedan",
            "Tesla_X_or_Y": "suv",
            "Other car": "unknown body type",
        },
        "gates": {
            "metadata_integrity_pass": True,
            "metadata_schema_pass": True,
            "license_allows_research_reuse_with_attribution": True,
            "image_archive_downloaded": False,
            "image_integrity_audited": False,
            "vehicle_foreground_quality_audited": False,
            "duplicate_and_vehicle_group_audited": False,
            "training_authorized": False,
        },
        "scope": {
            "image_pixels_opened": 0,
            "validation_or_test_images_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.report), ensure_ascii=False))


if __name__ == "__main__":
    main()
