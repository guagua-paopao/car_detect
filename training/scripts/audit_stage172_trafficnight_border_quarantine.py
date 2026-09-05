from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


OFFICIAL_CLASSES = {
    "Car",
    "Buses",
    "Trucks",
    "Tractor",
    "Semi-Trailers",
    "Empty Semi-Trailers",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"drive-qualified ZIP member path: {name!r}")
    return str(path)


def load_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def audit_border_quarantine(
    archive: Path,
    source_audit: Path,
    *,
    expected_archive_sha256: str,
    max_quarantine_fraction: float,
    minimum_retained_objects: int,
    minimum_retained_images: int,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not source_audit.is_file():
        raise FileNotFoundError(source_audit)
    if not 0.0 <= max_quarantine_fraction < 1.0:
        raise ValueError("max_quarantine_fraction must be in [0, 1)")

    archive_sha256 = sha256_file(archive)
    source = load_json(source_audit)
    failures: list[str] = []
    expected_sha = expected_archive_sha256.lower()
    if archive_sha256 != expected_sha:
        failures.append(f"archive SHA256 mismatch: {archive_sha256} != {expected_sha}")
    if str(source.get("archive_sha256", "")).lower() != archive_sha256:
        failures.append("source audit archive SHA256 mismatch")
    source_invalid_shapes = source.get("invalid_shapes")
    expected_source_failures = [f"invalid shapes: {source_invalid_shapes}"]
    if source.get("status") != "fail" or source.get("failures") != expected_source_failures:
        failures.append("source audit did not fail exclusively on invalid shapes")
    for key in (
        "crc_verified",
        "invalid_images",
        "invalid_json",
        "unknown_class_counts",
        "images_without_labels",
        "labels_without_images",
        "duplicate_normalized_names",
        "frozen_markers",
    ):
        value = source.get(key)
        if key == "crc_verified":
            if value is not True:
                failures.append("source audit CRC is not verified")
        elif value not in (0, {}, []):
            failures.append(f"source audit has an unrelated failure signal: {key}={value!r}")

    retained_by_class: Counter[str] = Counter()
    quarantined_by_class: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    quarantine_objects: list[dict[str, object]] = []
    affected_images: set[str] = set()
    retained_images: set[str] = set()
    images_with_shapes: set[str] = set()
    structural_failures = 0

    with zipfile.ZipFile(archive, "r") as opened:
        for info in opened.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            try:
                member = normalized_member_name(info.filename)
                document = json.loads(opened.read(info).decode("utf-8-sig"))
            except Exception as error:
                failures.append(f"unreadable annotation member: {info.filename}: {error}")
                continue
            if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
                failures.append(f"invalid LabelMe document: {member}")
                continue
            width = document.get("imageWidth")
            height = document.get("imageHeight")
            if not (
                isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and width > 0
                and height > 0
            ):
                failures.append(f"invalid image dimensions: {member}")
                continue
            if document["shapes"]:
                images_with_shapes.add(member)
            retained_in_image = 0
            for index, shape in enumerate(document["shapes"]):
                if not isinstance(shape, dict):
                    structural_failures += 1
                    continue
                label = str(shape.get("label", "")).strip()
                points = shape.get("points")
                shape_type = str(shape.get("shape_type", "")).strip().lower()
                if label not in OFFICIAL_CLASSES:
                    structural_failures += 1
                    continue
                if not isinstance(points, list) or len(points) != 4:
                    structural_failures += 1
                    continue
                numeric_points: list[tuple[float, float]] = []
                valid_structure = True
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        valid_structure = False
                        break
                    x, y = point
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        valid_structure = False
                        break
                    x_value = float(x)
                    y_value = float(y)
                    if not math.isfinite(x_value) or not math.isfinite(y_value):
                        valid_structure = False
                        break
                    numeric_points.append((x_value, y_value))
                if not valid_structure:
                    structural_failures += 1
                    continue

                reason = None
                if shape_type != "polygon":
                    reason = f"unsupported_shape_type:{shape_type or '<missing>'}"
                elif any(
                    x < 0.0 or x > float(width) or y < 0.0 or y > float(height)
                    for x, y in numeric_points
                ):
                    reason = "out_of_bounds"
                if reason is None:
                    retained_by_class[label] += 1
                    retained_in_image += 1
                    continue

                quarantined_by_class[label] += 1
                quarantine_reasons[reason] += 1
                affected_images.add(member)
                quarantine_objects.append(
                    {
                        "annotation_member": member,
                        "shape_index": index,
                        "label": label,
                        "reason": reason,
                        "shape_type": shape_type,
                        "points": [[x, y] for x, y in numeric_points],
                        "image_width": int(width),
                        "image_height": int(height),
                    }
                )
            if retained_in_image:
                retained_images.add(member)

    if structural_failures:
        failures.append(f"structurally invalid shapes cannot be quarantined: {structural_failures}")
    retained_objects = sum(retained_by_class.values())
    quarantined_objects = sum(quarantined_by_class.values())
    total_objects = retained_objects + quarantined_objects + structural_failures
    quarantine_fraction = quarantined_objects / total_objects if total_objects else 1.0
    if quarantine_fraction > max_quarantine_fraction:
        failures.append(
            f"quarantine fraction exceeds limit: {quarantine_fraction:.8f} > {max_quarantine_fraction:.8f}"
        )
    if retained_objects < minimum_retained_objects:
        failures.append(f"retained objects below minimum: {retained_objects} < {minimum_retained_objects}")
    if len(retained_images) < minimum_retained_images:
        failures.append(f"retained images below minimum: {len(retained_images)} < {minimum_retained_images}")

    return {
        "schema_version": "stage172-trafficnight-border-quarantine-v1",
        "status": "pass_quarantine_plan" if not failures else "fail",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "source_audit": str(source_audit),
        "source_audit_sha256": sha256_file(source_audit),
        "policy": "discard complete out-of-bounds or unsupported shapes; never clip coordinates or fabricate geometry",
        "retained_objects": retained_objects,
        "quarantined_objects": quarantined_objects,
        "structural_failures": structural_failures,
        "total_objects": total_objects,
        "quarantine_fraction": quarantine_fraction,
        "maximum_quarantine_fraction": max_quarantine_fraction,
        "retained_images": len(retained_images),
        "images_with_shapes": len(images_with_shapes),
        "affected_images": len(affected_images),
        "images_with_zero_retained_objects": len(images_with_shapes - retained_images),
        "retained_class_counts": dict(sorted(retained_by_class.items())),
        "quarantined_class_counts": dict(sorted(quarantined_by_class.items())),
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
        "quarantine_objects": quarantine_objects,
        "images_extracted": False,
        "training_started": False,
        "test_payload_opened": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fail-closed TrafficNight border quarantine plan.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-quarantine-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-retained-objects", type=int, default=39000)
    parser.add_argument("--minimum-retained-images", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_border_quarantine(
        args.archive.resolve(),
        args.source_audit.resolve(),
        expected_archive_sha256=args.expected_archive_sha256,
        max_quarantine_fraction=args.max_quarantine_fraction,
        minimum_retained_objects=args.minimum_retained_objects,
        minimum_retained_images=args.minimum_retained_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass_quarantine_plan" else 1


if __name__ == "__main__":
    raise SystemExit(main())
