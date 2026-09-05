#!/usr/bin/env python3
"""Audit LabelMe-to-decoded-image dimension transforms before TrafficNight extraction."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_member_name(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise ValueError(f"unsafe ZIP member: {name}")
    return str(path)


def scale_points(
    points: object,
    declared_width: float,
    declared_height: float,
    decoded_width: int,
    decoded_height: int,
) -> list[tuple[float, float]]:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("invalid polygon")
    scale_x = decoded_width / declared_width
    scale_y = decoded_height / declared_height
    result: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite point")
        result.append((x * scale_x, y * scale_y))
    return result


def audit(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    quarantine_report = args.quarantine_report.resolve()
    actual_archive_sha = sha256_file(archive)
    actual_report_sha = sha256_file(quarantine_report)
    failures: list[str] = []
    if actual_archive_sha != args.expected_archive_sha256.lower():
        failures.append("archive SHA256 mismatch")
    if actual_report_sha != args.expected_quarantine_report_sha256.lower():
        failures.append("quarantine report SHA256 mismatch")
    quarantine_document = json.loads(quarantine_report.read_text(encoding="utf-8"))
    if quarantine_document.get("status") != "pass_quarantine_plan":
        failures.append("quarantine plan did not pass")
    quarantine = {
        (str(item["annotation_member"]), int(item["shape_index"]))
        for item in quarantine_document.get("quarantine_objects", [])
    }

    pair_counts: Counter[str] = Counter()
    transform_counts: Counter[str] = Counter()
    retained_shapes = 0
    transformed_retained_shapes = 0
    transformed_out_of_bounds = 0
    observed_quarantine: set[tuple[str, int]] = set()
    examples: list[dict[str, object]] = []

    with zipfile.ZipFile(archive) as opened:
        images: dict[str, str] = {}
        annotations: dict[str, str] = {}
        for raw_name in opened.namelist():
            name = normalized_member_name(raw_name)
            suffix = PurePosixPath(name).suffix.lower()
            stem = str(PurePosixPath(name).with_suffix(""))
            if suffix in {".jpg", ".jpeg", ".png"}:
                images[stem] = name
            elif suffix == ".json":
                annotations[stem] = name
        if set(images) != set(annotations):
            failures.append("image/annotation pairing drift")
        for stem in sorted(set(images) & set(annotations)):
            document = json.loads(opened.read(annotations[stem]).decode("utf-8-sig"))
            declared_width = float(document.get("imageWidth", 0))
            declared_height = float(document.get("imageHeight", 0))
            if declared_width <= 0 or declared_height <= 0:
                failures.append(f"invalid declared dimensions: {annotations[stem]}")
                continue
            with Image.open(io.BytesIO(opened.read(images[stem]))) as image:
                decoded_width, decoded_height = image.size
            pattern = f"{int(declared_width)}x{int(declared_height)}->{decoded_width}x{decoded_height}"
            pair_counts[pattern] += 1
            transformed = declared_width != decoded_width or declared_height != decoded_height
            transform_counts["images_requiring_declared_to_decoded_scale" if transformed else "images_identity"] += 1
            if transformed and len(examples) < 20:
                examples.append({
                    "annotation_member": annotations[stem],
                    "declared_dimensions": [int(declared_width), int(declared_height)],
                    "decoded_dimensions": [decoded_width, decoded_height],
                    "scale_x": decoded_width / declared_width,
                    "scale_y": decoded_height / declared_height,
                })
            for shape_index, shape in enumerate(document.get("shapes", [])):
                key = (annotations[stem], shape_index)
                if key in quarantine:
                    observed_quarantine.add(key)
                    continue
                retained_shapes += 1
                try:
                    points = scale_points(
                        shape.get("points"), declared_width, declared_height,
                        decoded_width, decoded_height,
                    )
                except Exception as error:
                    failures.append(f"unscalable retained shape: {annotations[stem]}#{shape_index}: {error}")
                    continue
                if transformed:
                    transformed_retained_shapes += 1
                if any(
                    x < 0 or x > decoded_width or y < 0 or y > decoded_height
                    for x, y in points
                ):
                    transformed_out_of_bounds += 1

    if len(pair_counts) == 0:
        failures.append("no image pairs audited")
    if observed_quarantine != quarantine:
        failures.append(
            f"observed quarantine {len(observed_quarantine)} != expected {len(quarantine)}"
        )
    if retained_shapes != int(quarantine_document.get("retained_objects", -1)):
        failures.append(
            f"retained shapes {retained_shapes} != audited {quarantine_document.get('retained_objects')}"
        )
    if transformed_out_of_bounds:
        failures.append(f"scaled retained shapes out of bounds: {transformed_out_of_bounds}")

    return {
        "schema_version": "stage172-trafficnight-dimension-alignment-v1",
        "status": "pass_affine_scale_plan" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_sha256": actual_archive_sha,
        "quarantine_report": str(quarantine_report),
        "quarantine_report_sha256": actual_report_sha,
        "paired_images": sum(pair_counts.values()),
        "dimension_patterns": dict(sorted(pair_counts.items())),
        "transform_counts": dict(sorted(transform_counts.items())),
        "retained_shapes": retained_shapes,
        "transformed_retained_shapes": transformed_retained_shapes,
        "scaled_retained_shapes_out_of_bounds": transformed_out_of_bounds,
        "quarantine_keys_observed": len(observed_quarantine),
        "transform_examples": examples,
        "policy": {
            "transform": "multiply each LabelMe x/y by decoded_width/declared_width and decoded_height/declared_height",
            "coordinate_clipping": False,
            "coordinate_repair": False,
            "geometry_fabricated": False,
            "quarantined_shapes_transformed_or_used": False,
            "images_extracted": False,
            "training_started": False,
            "test_payload_opened": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--quarantine-report", type=Path, required=True)
    parser.add_argument("--expected-quarantine-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "paired_images": report["paired_images"],
        "transformed_retained_shapes": report["transformed_retained_shapes"],
        "failures": report["failures"],
    }, ensure_ascii=False))
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
