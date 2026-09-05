#!/usr/bin/env python3
"""Build conservative, train-only TrafficNight crops with lineage-wide deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from build_stage70_specialist_manifests import HammingBKTree


SOURCE_MAPPING = {
    "Buses": {"body_type": "bus", "coarse_body_family": "", "fine_supervised": True},
    "Trucks": {"body_type": "truck", "coarse_body_family": "truck", "fine_supervised": True},
    "Tractor": {"body_type": "unknown", "coarse_body_family": "truck", "fine_supervised": False},
    "Car": {"body_type": "unknown", "coarse_body_family": "car", "fine_supervised": False},
    "Semi-Trailers": {"body_type": "unknown", "coarse_body_family": "", "fine_supervised": False},
    "Empty Semi-Trailers": {"body_type": "unknown", "coarse_body_family": "", "fine_supervised": False},
}

FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "color", "color_supervised", "source_dataset", "source_label", "source_license",
    "source_license_url", "license_train_eligible", "review_status", "review_method",
    "camera_id", "video_id", "track_group", "source_frame_id", "source_box_index",
    "crop_quality", "viewpoint", "lighting", "night", "occluded", "truncated",
    "blur", "width", "height", "gray_mean", "gray_stddev", "edge_variance",
    "sha256", "dhash64", "adverse_supervised", "formal_train_eligible",
    "research_only", "deployment_eligible", "stage172_origin", "stage172_polygon_points",
    "stage172_annotation_member", "stage172_dedup_policy", "stage172_source_polygon_points",
    "stage172_declared_dimensions", "stage172_decoded_dimensions", "stage172_dimension_transform",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def load_base(path: Path) -> tuple[set[str], HammingBKTree, int, Counter[str]]:
    exact: set[str] = set()
    tree = HammingBKTree()
    rows = 0
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            digest = content_hash(row)
            if digest:
                exact.add(digest)
                counts["base_exact_hashes"] += 1
            value = perceptual_hash(row)
            if len(value) == 16:
                try:
                    tree.add(int(value, 16), row)
                    counts["base_perceptual_hashes"] += 1
                except ValueError:
                    counts["base_invalid_perceptual_hashes"] += 1
    return exact, tree, rows, counts


def normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return str(path)


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if min(values.shape) > 4:
        values = values[2:-2, 2:-2]
    return float(values.var())


def inspect_crop(image: Image.Image) -> dict[str, object]:
    width, height = image.size
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    usable = min(width, height) >= 16 and stddev >= 5 and edges >= 10 and 2 <= mean <= 253
    fine_supervision = min(width, height) >= 24 and stddev >= 8 and edges >= 20 and 6 <= mean <= 249
    quality = "good" if min(width, height) >= 64 and stddev >= 18 and edges >= 60 else "small_or_hard"
    return {
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "usable": usable,
        "fine_supervision": fine_supervision,
        "crop_quality": quality,
        "blur": edges < 60,
    }


def validate_polygon(points: object, image_width: int, image_height: int) -> list[tuple[float, float]]:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("invalid_polygon_point_count")
    normalized: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid_polygon_point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite_polygon_point")
        if x < 0 or y < 0 or x > image_width or y > image_height:
            raise ValueError("out_of_bounds_polygon")
        normalized.append((x, y))
    area_twice = abs(sum(
        normalized[index][0] * normalized[(index + 1) % len(normalized)][1]
        - normalized[(index + 1) % len(normalized)][0] * normalized[index][1]
        for index in range(len(normalized))
    ))
    if area_twice < 2:
        raise ValueError("degenerate_polygon")
    return normalized


def scale_declared_points(
    points: object,
    declared_width: float,
    declared_height: float,
    decoded_width: int,
    decoded_height: int,
) -> list[list[float]]:
    if declared_width <= 0 or declared_height <= 0:
        raise ValueError("invalid_declared_dimensions")
    if not isinstance(points, list):
        raise ValueError("invalid_polygon_points")
    scale_x = decoded_width / declared_width
    scale_y = decoded_height / declared_height
    result: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid_polygon_point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite_polygon_point")
        result.append([x * scale_x, y * scale_y])
    return result


def polygon_crop(image: Image.Image, points: list[tuple[float, float]], padding: float) -> Image.Image:
    image_width, image_height = image.size
    left, right = min(point[0] for point in points), max(point[0] for point in points)
    top, bottom = min(point[1] for point in points), max(point[1] for point in points)
    width, height = right - left, bottom - top
    if width < 2 or height < 2:
        raise ValueError("degenerate_polygon_bounds")
    # All source points were already proven in bounds. Padding may be clamped, but
    # source coordinates are never clipped or repaired.
    crop_left = max(0, int(math.floor(left - width * padding)))
    crop_top = max(0, int(math.floor(top - height * padding)))
    crop_right = min(image_width, int(math.ceil(right + width * padding)) + 1)
    crop_bottom = min(image_height, int(math.ceil(bottom + height * padding)) + 1)
    if crop_right - crop_left < 8 or crop_bottom - crop_top < 8:
        raise ValueError("crop_dimension_below_8")
    return image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")


def encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, optimize=True)
    return output.getvalue()


def quarantine_keys(report: dict[str, object]) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for item in report.get("quarantine_objects", []):
        if not isinstance(item, dict):
            raise RuntimeError("invalid quarantine object")
        key = (str(item.get("annotation_member", "")), int(item.get("shape_index", -1)))
        if not key[0] or key[1] < 0 or key in result:
            raise RuntimeError("invalid or duplicate quarantine key")
        result.add(key)
    return result


def pair_members(names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    annotations: dict[str, str] = {}
    for raw_name in names:
        name = normalized_member_name(raw_name)
        suffix = PurePosixPath(name).suffix.lower()
        stem = str(PurePosixPath(name).with_suffix(""))
        if suffix in {".jpg", ".jpeg", ".png"}:
            if stem in images:
                raise RuntimeError(f"duplicate image stem: {stem}")
            images[stem] = name
        elif suffix == ".json":
            if stem in annotations:
                raise RuntimeError(f"duplicate annotation stem: {stem}")
            annotations[stem] = name
    return images, annotations


def build(args: argparse.Namespace) -> dict[str, object]:
    inputs = {
        "archive": args.archive.resolve(),
        "quarantine_report": args.quarantine_report.resolve(),
        "dimension_report": args.dimension_report.resolve(),
        "base_manifest": args.base_manifest.resolve(),
        "labels": args.labels.resolve(),
        "dedup_library": args.dedup_library.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {key: sha256_file(path) for key, path in inputs.items()}
    expected = {
        "archive": args.expected_archive_sha256.lower(),
        "quarantine_report": args.expected_quarantine_report_sha256.lower(),
        "dimension_report": args.expected_dimension_report_sha256.lower(),
        "base_manifest": args.expected_base_manifest_sha256.lower(),
        "labels": args.expected_labels_sha256.lower(),
        "dedup_library": args.expected_dedup_library_sha256.lower(),
    }
    if actual != expected:
        raise RuntimeError(f"pinned input SHA mismatch: actual={actual}, expected={expected}")

    audit = json.loads(inputs["quarantine_report"].read_text(encoding="utf-8"))
    if audit.get("status") != "pass_quarantine_plan" or audit.get("failures"):
        raise RuntimeError("TrafficNight quarantine audit did not pass")
    if any(bool(audit.get(key)) for key in (
        "images_extracted", "training_started", "test_payload_opened", "frozen_video_used",
        "production_model_modified", "deployment_performed",
    )):
        raise RuntimeError("TrafficNight quarantine audit violates access or production policy")
    if str(audit.get("archive_sha256", "")).lower() != actual["archive"]:
        raise RuntimeError("quarantine report archive lineage mismatch")
    quarantine = quarantine_keys(audit)
    if len(quarantine) != int(audit.get("quarantined_objects", -1)):
        raise RuntimeError("quarantine object count mismatch")

    dimension_audit = json.loads(inputs["dimension_report"].read_text(encoding="utf-8"))
    if dimension_audit.get("status") != "pass_affine_scale_plan" or dimension_audit.get("failures"):
        raise RuntimeError("TrafficNight dimension-alignment audit did not pass")
    if str(dimension_audit.get("archive_sha256", "")).lower() != actual["archive"]:
        raise RuntimeError("dimension report archive lineage mismatch")
    if str(dimension_audit.get("quarantine_report_sha256", "")).lower() != actual["quarantine_report"]:
        raise RuntimeError("dimension report quarantine lineage mismatch")

    contract = json.loads(inputs["labels"].read_text(encoding="utf-8"))
    required = {"unknown", "truck", "bus"}
    if not required.issubset(set(contract.get("body_types", []))):
        raise RuntimeError("vehicle labels contract cannot represent conservative mapping")

    output_images = args.output_images.resolve()
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError("refusing to use non-empty Stage172 crop directory")
    output_images.mkdir(parents=True, exist_ok=True)
    initial_free_bytes = shutil.disk_usage(output_images).free
    if initial_free_bytes < args.minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free space: {initial_free_bytes} < {args.minimum_free_bytes}"
        )

    base_exact, base_tree, base_rows, base_counts = load_base(inputs["base_manifest"])
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter(base_counts)
    source_counts: Counter[str] = Counter()
    retained_source_counts: Counter[str] = Counter()
    fine_counts: Counter[str] = Counter()
    coarse_counts: Counter[str] = Counter()
    observed_quarantine: set[tuple[str, int]] = set()
    output_bytes = 0

    with zipfile.ZipFile(inputs["archive"]) as opened:
        if opened.testzip() is not None:
            raise RuntimeError("TrafficNight archive CRC failure")
        images, annotations = pair_members(opened.namelist())
        if set(images) != set(annotations) or len(images) != int(audit.get("images_with_shapes", -1)):
            raise RuntimeError("TrafficNight image/annotation pairing drift")

        for stem in sorted(images):
            image_payload = opened.read(images[stem])
            annotation_payload = opened.read(annotations[stem])
            counters["image_payloads_read"] += 1
            counters["annotation_payloads_read"] += 1
            with Image.open(io.BytesIO(image_payload)) as source:
                source.load()
                frame = source.convert("RGB")
            annotation = json.loads(annotation_payload.decode("utf-8-sig", errors="strict"))
            declared_width = float(annotation.get("imageWidth", 0))
            declared_height = float(annotation.get("imageHeight", 0))
            if declared_width <= 0 or declared_height <= 0:
                raise RuntimeError(f"invalid annotation dimensions: {annotations[stem]}")
            dimension_transform = declared_width != frame.width or declared_height != frame.height
            if dimension_transform:
                counters["images_with_declared_to_decoded_affine_scale"] += 1
            if PurePosixPath(str(annotation.get("imagePath", ""))).name.lower() != PurePosixPath(images[stem]).name.lower():
                raise RuntimeError(f"annotation imagePath differs: {annotations[stem]}")
            shapes = annotation.get("shapes")
            if not isinstance(shapes, list):
                raise RuntimeError(f"invalid shapes collection: {annotations[stem]}")

            frame_id = PurePosixPath(images[stem]).stem
            date_token = frame_id.split("_")[1][:8] if "_" in frame_id else "unknown"
            for shape_index, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    raise RuntimeError(f"invalid shape: {annotations[stem]}#{shape_index}")
                source_label = str(shape.get("label", ""))
                source_counts[source_label] += 1
                key = (annotations[stem], shape_index)
                if key in quarantine:
                    observed_quarantine.add(key)
                    counters["quarantined_source_objects_skipped"] += 1
                    continue
                if source_label not in SOURCE_MAPPING:
                    raise RuntimeError(f"unknown source label: {source_label}")
                if shape.get("shape_type") != "polygon":
                    raise RuntimeError(f"unquarantined unsupported shape: {annotations[stem]}#{shape_index}")
                try:
                    source_points = shape.get("points")
                    scaled_points = scale_declared_points(
                        source_points, declared_width, declared_height, frame.width, frame.height,
                    )
                    points = validate_polygon(scaled_points, frame.width, frame.height)
                    crop = polygon_crop(frame, points, args.padding)
                    metrics = inspect_crop(crop)
                except Exception as exc:
                    counters[f"rejected_crop:{exc}"] += 1
                    continue
                if not bool(metrics["usable"]):
                    counters["rejected_unusable_crop"] += 1
                    continue

                mapping = SOURCE_MAPPING[source_label]
                fine_supervised = bool(mapping["fine_supervised"] and metrics["fine_supervision"])
                body_type = str(mapping["body_type"]) if fine_supervised else "unknown"
                coarse_family = str(mapping["coarse_body_family"])
                if mapping["fine_supervised"] and not fine_supervised:
                    counters[f"demoted_low_quality_fine_supervision:{mapping['body_type']}"] += 1

                encoded = encode_jpeg(crop)
                # Deduplication and manifest hashes must describe the persisted
                # JPEG, not the pre-encoding pixel buffer. JPEG quantization can
                # otherwise change a few dHash bits and reintroduce near duplicates.
                with Image.open(io.BytesIO(encoded)) as persisted:
                    persisted.load()
                    crop_dhash = dhash64(persisted)
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in base_exact or digest in candidate_exact:
                    counters["rejected_exact_duplicate_crop"] += 1
                    continue
                if base_tree.find(crop_dhash, args.near_duplicate_hamming) is not None:
                    counters["rejected_base_near_duplicate_crop"] += 1
                    continue
                near = candidate_tree.find(crop_dhash, args.near_duplicate_hamming)
                if near is not None:
                    counters["rejected_candidate_near_duplicate_crop"] += 1
                    if near.get("source_label") != source_label:
                        counters["rejected_near_duplicate_label_conflict"] += 1
                    continue

                destination = output_images / f"trafficnight_{frame_id}_{shape_index:03d}.jpg"
                if output_bytes + len(encoded) > args.maximum_output_bytes:
                    raise RuntimeError(
                        f"output byte guard exceeded: {output_bytes + len(encoded)} > {args.maximum_output_bytes}"
                    )
                if len(rows) % 250 == 0 and shutil.disk_usage(output_images).free < args.minimum_free_bytes:
                    raise RuntimeError("free-space guard tripped during Stage172 extraction")
                destination.write_bytes(encoded)
                output_bytes += len(encoded)
                row = {
                    "image_path": str(destination),
                    "split": "train",
                    "body_type": body_type,
                    "body_type_supervised": str(fine_supervised).lower(),
                    "coarse_body_family": coarse_family,
                    "color": "unknown",
                    "color_supervised": "false",
                    "source_dataset": "TrafficNight-sRGB-ImgLabel",
                    "source_label": source_label,
                    "source_license": "Apache-2.0",
                    "source_license_url": "https://github.com/ChenhongyiYang/TrafficNight",
                    "license_train_eligible": "true",
                    "review_status": "approved_official_polygon_quality_gate" if fine_supervised else "approved_coarse_or_unlabeled_consistency_only",
                    "review_method": "official_polygon+border_quarantine+decode+quality_gate+sha256+dhash4_full_stage167_lineage",
                    "camera_id": "trafficnight_aerial_camera_unknown",
                    "video_id": f"trafficnight_{date_token}",
                    "track_group": f"trafficnight_frame_{frame_id}",
                    "source_frame_id": frame_id,
                    "source_box_index": str(shape_index),
                    "crop_quality": str(metrics["crop_quality"]),
                    "viewpoint": "aerial_oblique_night_auxiliary",
                    "lighting": "night_source_truth",
                    "night": "true",
                    "occluded": "unknown",
                    "truncated": "false",
                    "blur": str(metrics["blur"]).lower(),
                    "width": str(metrics["width"]),
                    "height": str(metrics["height"]),
                    "gray_mean": str(metrics["gray_mean"]),
                    "gray_stddev": str(metrics["gray_stddev"]),
                    "edge_variance": str(metrics["edge_variance"]),
                    "sha256": digest,
                    "dhash64": f"{crop_dhash:016x}",
                    "adverse_supervised": "true",
                    "formal_train_eligible": "true",
                    "research_only": "false",
                    "deployment_eligible": "false",
                    "stage172_origin": "trafficnight_safe_border_quarantine",
                    "stage172_polygon_points": json.dumps(points, separators=(",", ":")),
                    "stage172_source_polygon_points": json.dumps(source_points, separators=(",", ":")),
                    "stage172_declared_dimensions": f"{int(declared_width)}x{int(declared_height)}",
                    "stage172_decoded_dimensions": f"{frame.width}x{frame.height}",
                    "stage172_dimension_transform": "declared_to_decoded_affine_scale" if dimension_transform else "identity",
                    "stage172_annotation_member": annotations[stem],
                    "stage172_dedup_policy": f"exact+dhash<={args.near_duplicate_hamming}_against_stage167_and_candidate",
                }
                rows.append(row)
                candidate_exact.add(digest)
                candidate_tree.add(crop_dhash, row)
                retained_source_counts[source_label] += 1
                if fine_supervised:
                    fine_counts[body_type] += 1
                if coarse_family:
                    coarse_counts[coarse_family] += 1

    failures: list[str] = []
    if observed_quarantine != quarantine:
        failures.append(f"observed quarantine keys {len(observed_quarantine)} != audited {len(quarantine)}")
    if sum(source_counts.values()) != int(audit.get("total_objects", -1)):
        failures.append(f"source objects {sum(source_counts.values())} != audited {audit.get('total_objects')}")
    if len(rows) < args.minimum_rows:
        failures.append(f"retained rows {len(rows)} < {args.minimum_rows}")
    if sum(fine_counts.values()) < args.minimum_fine_supervised_rows:
        failures.append(f"fine supervised rows {sum(fine_counts.values())} < {args.minimum_fine_supervised_rows}")
    if fine_counts["bus"] < args.minimum_bus_rows:
        failures.append(f"bus rows {fine_counts['bus']} < {args.minimum_bus_rows}")
    if fine_counts["truck"] < args.minimum_truck_rows:
        failures.append(f"truck rows {fine_counts['truck']} < {args.minimum_truck_rows}")
    if sum(coarse_counts.values()) < args.minimum_coarse_rows:
        failures.append(f"coarse rows {sum(coarse_counts.values())} < {args.minimum_coarse_rows}")

    report = {
        "schema_version": "stage172-trafficnight-train-crops-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            key: {"path": str(path), "sha256": actual[key]} for key, path in inputs.items()
        } | {"base_manifest_rows": base_rows},
        "source_mapping": SOURCE_MAPPING,
        "output": {
            "images": str(output_images),
            "image_bytes": output_bytes,
            "rows": len(rows),
            "fine_supervised_rows": sum(fine_counts.values()),
            "coarse_rows": sum(coarse_counts.values()),
            "source_counts": dict(sorted(source_counts.items())),
            "retained_source_counts": dict(sorted(retained_source_counts.items())),
            "fine_body_counts": dict(sorted(fine_counts.items())),
            "coarse_family_counts": dict(sorted(coarse_counts.items())),
            "counters": dict(sorted(counters.items())),
        },
        "policy": {
            "source_validation_payloads_read": 0,
            "source_test_payloads_read": 0,
            "validation_rows_created": 0,
            "test_rows_created": 0,
            "all_outputs_train_only": True,
            "initial_free_bytes": initial_free_bytes,
            "minimum_free_bytes_guard": args.minimum_free_bytes,
            "maximum_output_bytes_guard": args.maximum_output_bytes,
            "source_coordinates_clipped_or_repaired": False,
            "declared_to_decoded_affine_scale_audited": True,
            "perceptual_hash_computed_from_persisted_jpeg": True,
            "quarantined_objects_skipped": len(observed_quarantine),
            "fine_labels_fabricated": False,
            "color_labels_fabricated": False,
            "all_colors_unknown": True,
            "aerial_data_replaces_gate_camera_validation": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
        "failures": failures,
    }
    return {"report": report, "rows": rows}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--quarantine-report", type=Path, required=True)
    parser.add_argument("--expected-quarantine-report-sha256", required=True)
    parser.add_argument("--dimension-report", type=Path, required=True)
    parser.add_argument("--expected-dimension-report-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--dedup-library", type=Path, required=True)
    parser.add_argument("--expected-dedup-library-sha256", required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-rows", type=int, default=5000)
    parser.add_argument("--minimum-fine-supervised-rows", type=int, default=500)
    parser.add_argument("--minimum-bus-rows", type=int, default=75)
    parser.add_argument("--minimum-truck-rows", type=int, default=250)
    parser.add_argument("--minimum-coarse-rows", type=int, default=2500)
    parser.add_argument("--minimum-free-bytes", type=int, default=100 * 1024 ** 3)
    parser.add_argument("--maximum-output-bytes", type=int, default=8 * 1024 ** 3)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage172 evidence")
    if not 0 <= args.near_duplicate_hamming <= 7 or not 0 <= args.padding <= 0.5:
        raise ValueError("invalid dedup radius or crop padding")
    if args.minimum_free_bytes < 0 or args.maximum_output_bytes <= 0:
        raise ValueError("invalid disk-space guard")
    result = build(args)
    report = result["report"]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        print(json.dumps({"status": "fail", "failures": report["failures"]}, ensure_ascii=False))
        raise SystemExit(1)
    write_manifest(args.output_manifest, result["rows"])
    report["output"].update({
        "manifest": str(args.output_manifest.resolve()),
        "manifest_sha256": sha256_file(args.output_manifest),
    })
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "rows": report["output"]["rows"],
        "fine_supervised_rows": report["output"]["fine_supervised_rows"],
        "coarse_rows": report["output"]["coarse_rows"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
