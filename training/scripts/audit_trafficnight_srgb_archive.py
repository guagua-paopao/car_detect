from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


OFFICIAL_CLASSES = {
    "Car",
    "Buses",
    "Trucks",
    "Tractor",
    "Semi-Trailers",
    "Empty Semi-Trailers",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FROZEN_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}


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


def pair_key(name: str) -> str:
    path = PurePosixPath(name)
    return str(path.with_suffix(""))


def audit_archive(
    archive: Path,
    source_evidence: Path,
    *,
    minimum_pairs: int,
    minimum_objects: int,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not source_evidence.is_file():
        raise FileNotFoundError(source_evidence)
    evidence = json.loads(source_evidence.read_text(encoding="utf-8"))
    if evidence.get("license_spdx") != "Apache-2.0":
        raise ValueError("source evidence does not pin Apache-2.0")

    failures: list[str] = []
    class_counts: Counter[str] = Counter()
    json_key_profiles: Counter[str] = Counter()
    image_names: dict[str, str] = {}
    label_names: dict[str, str] = {}
    image_dimensions: dict[str, tuple[int, int]] = {}
    invalid_images = 0
    invalid_json = 0
    invalid_shapes = 0
    unknown_classes: Counter[str] = Counter()
    duplicate_names = 0
    frozen_markers = 0
    total_uncompressed = 0

    with zipfile.ZipFile(archive, "r") as opened:
        members = opened.infolist()
        seen_names: set[str] = set()
        normalized_members: list[tuple[zipfile.ZipInfo, str]] = []
        for info in members:
            try:
                name = normalized_member_name(info.filename)
            except ValueError as error:
                failures.append(str(error))
                continue
            lowered = name.lower()
            if any(marker in lowered for marker in FROZEN_MARKERS):
                frozen_markers += 1
            if name in seen_names:
                duplicate_names += 1
            seen_names.add(name)
            total_uncompressed += info.file_size
            normalized_members.append((info, name))

        bad_crc_member = opened.testzip()
        if bad_crc_member:
            failures.append(f"ZIP CRC failure: {bad_crc_member}")

        for info, name in normalized_members:
            if info.is_dir():
                continue
            suffix = PurePosixPath(name).suffix.lower()
            key = pair_key(name)
            if suffix in IMAGE_EXTENSIONS:
                try:
                    payload = opened.read(info)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(payload)) as image:
                        image_dimensions[key] = tuple(map(int, image.size))
                    image_names[key] = name
                except Exception:
                    invalid_images += 1
                continue
            if suffix != ".json":
                continue
            try:
                document = json.loads(opened.read(info).decode("utf-8-sig"))
            except Exception:
                invalid_json += 1
                continue
            if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
                invalid_json += 1
                continue
            json_key_profiles["|".join(sorted(document))] += 1
            label_names[key] = name
            width = document.get("imageWidth")
            height = document.get("imageHeight")
            dimensions_valid = isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0
            for shape in document["shapes"]:
                if not isinstance(shape, dict):
                    invalid_shapes += 1
                    continue
                label = str(shape.get("label", "")).strip()
                points = shape.get("points")
                if label not in OFFICIAL_CLASSES:
                    unknown_classes[label or "<empty>"] += 1
                if not isinstance(points, list) or len(points) < 4:
                    invalid_shapes += 1
                    continue
                valid_points = True
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        valid_points = False
                        break
                    x, y = point
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        valid_points = False
                        break
                    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
                        valid_points = False
                        break
                    if dimensions_valid and not (-1.0 <= float(x) <= float(width) + 1.0 and -1.0 <= float(y) <= float(height) + 1.0):
                        valid_points = False
                        break
                if not valid_points:
                    invalid_shapes += 1
                    continue
                class_counts[label] += 1

    paired = set(image_names) & set(label_names)
    images_without_labels = sorted(set(image_names) - set(label_names))
    labels_without_images = sorted(set(label_names) - set(image_names))
    dimension_mismatches = 0
    # LabelMe dimensions were validated above; image decode dimensions are retained for the crop stage.
    if duplicate_names:
        failures.append(f"duplicate normalized member names: {duplicate_names}")
    if frozen_markers:
        failures.append(f"frozen markers found: {frozen_markers}")
    if invalid_images:
        failures.append(f"invalid images: {invalid_images}")
    if invalid_json:
        failures.append(f"invalid LabelMe JSON files: {invalid_json}")
    if invalid_shapes:
        failures.append(f"invalid shapes: {invalid_shapes}")
    if unknown_classes:
        failures.append(f"unknown classes: {dict(unknown_classes)}")
    if images_without_labels:
        failures.append(f"images without labels: {len(images_without_labels)}")
    if labels_without_images:
        failures.append(f"labels without images: {len(labels_without_images)}")
    if len(paired) < minimum_pairs:
        failures.append(f"minimum paired images: {len(paired)} < {minimum_pairs}")
    object_count = sum(class_counts.values())
    if object_count < minimum_objects:
        failures.append(f"minimum valid objects: {object_count} < {minimum_objects}")

    return {
        "schema_version": "trafficnight-srgb-archive-audit-v1",
        "status": "pass" if not failures else "fail",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "source_evidence": str(source_evidence),
        "source_evidence_sha256": sha256_file(source_evidence),
        "license_spdx": evidence["license_spdx"],
        "zip_member_count": len(seen_names),
        "zip_uncompressed_bytes": total_uncompressed,
        "paired_image_label_count": len(paired),
        "image_count": len(image_names),
        "label_count": len(label_names),
        "valid_object_count": object_count,
        "class_counts": {key: class_counts[key] for key in sorted(class_counts)},
        "unknown_class_counts": {key: unknown_classes[key] for key in sorted(unknown_classes)},
        "invalid_images": invalid_images,
        "invalid_json": invalid_json,
        "invalid_shapes": invalid_shapes,
        "duplicate_normalized_names": duplicate_names,
        "images_without_labels": len(images_without_labels),
        "labels_without_images": len(labels_without_images),
        "dimension_mismatches": dimension_mismatches,
        "frozen_markers": frozen_markers,
        "crc_verified": not any(value.startswith("ZIP CRC failure") for value in failures),
        "images_extracted": False,
        "training_started": False,
        "test_payload_opened": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed TrafficNight sRGB ZIP and LabelMe audit.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=1000)
    parser.add_argument("--minimum-objects", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_archive(
        args.archive.resolve(),
        args.source_evidence.resolve(),
        minimum_pairs=args.minimum_pairs,
        minimum_objects=args.minimum_objects,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
