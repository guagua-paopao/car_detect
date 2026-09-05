from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


TRAIN_ANNOTATIONS = (
    "M3OT/Annotations/1/rgb/train_cocoformat.json",
    "M3OT/Annotations/2/rgb/train_cocoformat.json",
)
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48")


def digest_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"drive-qualified ZIP member path: {name!r}")
    return str(path)


def expected_rgb_train_member(drone: str, file_name: str) -> str:
    path = PurePosixPath(file_name.replace("\\", "/"))
    parts = path.parts
    try:
        train_index = parts.index("train")
    except ValueError as error:
        raise ValueError(f"COCO file_name is not a train path: {file_name!r}") from error
    suffix = parts[train_index + 1 :]
    if len(suffix) != 4 or suffix[0] != drone or suffix[2] != "img1":
        raise ValueError(f"unexpected COCO train path: {file_name!r}")
    return str(PurePosixPath("M3OT", drone, "rgb", "train", suffix[1], suffix[2], suffix[3]))


def audit_archive(
    archive: Path,
    source_evidence: Path,
    *,
    expected_bytes: int,
    expected_md5: str,
    minimum_rgb_train_images: int,
    minimum_rgb_train_annotations: int,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not source_evidence.is_file():
        raise FileNotFoundError(source_evidence)

    evidence = json.loads(source_evidence.read_text(encoding="utf-8"))
    failures: list[str] = []
    license_name = str(evidence.get("license", {}).get("name", ""))
    if license_name != "CC BY 4.0":
        failures.append(f"source evidence license mismatch: {license_name!r}")
    evidence_archive = evidence.get("archive", {})
    if int(evidence_archive.get("bytes", -1)) != expected_bytes:
        failures.append("source evidence archive size does not match the pinned size")
    if str(evidence_archive.get("supplied_md5", "")).lower() != expected_md5.lower():
        failures.append("source evidence MD5 does not match the pinned MD5")
    if archive.stat().st_size != expected_bytes:
        failures.append(f"archive size mismatch: {archive.stat().st_size} != {expected_bytes}")

    actual_md5, actual_sha256 = digest_file(archive)
    if actual_md5.lower() != expected_md5.lower():
        failures.append(f"archive MD5 mismatch: {actual_md5} != {expected_md5.lower()}")

    member_counts: Counter[str] = Counter()
    member_bytes: Counter[str] = Counter()
    train_sequences: dict[str, set[str]] = {"1": set(), "2": set()}
    duplicate_names = 0
    unsafe_paths = 0
    encrypted_members = 0
    symlink_members = 0
    frozen_markers = 0
    unsupported_top_level = 0
    normalized_names: set[str] = set()
    opened_members: list[str] = []
    coco_image_count = 0
    coco_annotation_count = 0
    coco_small_annotation_count = 0
    coco_invalid_images = 0
    coco_invalid_annotations = 0
    coco_missing_image_members = 0
    coco_duplicate_image_members = 0
    coco_track_keys: set[tuple[str, int, int]] = set()
    mapped_coco_members: set[str] = set()

    with zipfile.ZipFile(archive, "r") as opened:
        infos = opened.infolist()
        for info in infos:
            try:
                name = normalized_member_name(info.filename)
            except ValueError:
                unsafe_paths += 1
                continue
            if name in normalized_names:
                duplicate_names += 1
            normalized_names.add(name)
            if any(marker in name.lower() for marker in FROZEN_MARKERS):
                frozen_markers += 1
            if info.flag_bits & 0x1:
                encrypted_members += 1
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                symlink_members += 1
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != "M3OT":
                unsupported_top_level += 1
            if len(parts) >= 7 and parts[1] in {"1", "2"} and parts[2] in {"rgb", "ir"} and parts[3] in {"train", "val", "test"}:
                key = f"{parts[2]}_{parts[3]}"
                if not info.is_dir() and PurePosixPath(name).suffix.lower() == ".png":
                    member_counts[key] += 1
                    member_bytes[key] += info.file_size
                    if parts[2] == "rgb" and parts[3] == "train":
                        train_sequences[parts[1]].add(parts[4])

        for annotation_name in TRAIN_ANNOTATIONS:
            if annotation_name not in normalized_names:
                failures.append(f"missing train annotation: {annotation_name}")
                continue
            payload = opened.read(annotation_name)
            opened_members.append(annotation_name)
            try:
                document = json.loads(payload.decode("utf-8-sig"))
            except Exception as error:
                failures.append(f"invalid train COCO JSON {annotation_name}: {error}")
                continue
            drone = PurePosixPath(annotation_name).parts[2]
            categories = document.get("categories")
            if categories != [{"id": 1, "name": "vehicle"}]:
                failures.append(f"unexpected COCO categories in {annotation_name}: {categories!r}")
            images = document.get("images")
            annotations = document.get("annotations")
            if not isinstance(images, list) or not isinstance(annotations, list):
                failures.append(f"missing COCO images/annotations arrays: {annotation_name}")
                continue
            image_by_id: dict[int, dict[str, object]] = {}
            for image in images:
                if not isinstance(image, dict):
                    coco_invalid_images += 1
                    continue
                try:
                    image_id = int(image["id"])
                    width = int(image["width"])
                    height = int(image["height"])
                    member = expected_rgb_train_member(drone, str(image["file_name"]))
                except Exception:
                    coco_invalid_images += 1
                    continue
                if width <= 0 or height <= 0 or image_id in image_by_id:
                    coco_invalid_images += 1
                    continue
                image_by_id[image_id] = image
                if member in mapped_coco_members:
                    coco_duplicate_image_members += 1
                mapped_coco_members.add(member)
                if member not in normalized_names:
                    coco_missing_image_members += 1
            coco_image_count += len(image_by_id)

            seen_annotation_ids: set[int] = set()
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    coco_invalid_annotations += 1
                    continue
                try:
                    annotation_id = int(annotation["id"])
                    image_id = int(annotation["image_id"])
                    category_id = int(annotation["category_id"])
                    instance_id = int(annotation["instance_id"])
                    bbox = annotation["bbox"]
                    image = image_by_id[image_id]
                    width = float(image["width"])
                    height = float(image["height"])
                    video_id = int(image["video_id"])
                    x, y, box_width, box_height = map(float, bbox)
                except Exception:
                    coco_invalid_annotations += 1
                    continue
                finite = all(math.isfinite(value) for value in (x, y, box_width, box_height))
                in_bounds = x >= 0 and y >= 0 and box_width > 0 and box_height > 0 and x + box_width <= width + 1 and y + box_height <= height + 1
                if annotation_id in seen_annotation_ids or category_id != 1 or not finite or not in_bounds:
                    coco_invalid_annotations += 1
                    continue
                seen_annotation_ids.add(annotation_id)
                coco_annotation_count += 1
                if box_width * box_height < 1024:
                    coco_small_annotation_count += 1
                coco_track_keys.add((drone, video_id, instance_id))

    if duplicate_names:
        failures.append(f"duplicate normalized member names: {duplicate_names}")
    if unsafe_paths:
        failures.append(f"unsafe member paths: {unsafe_paths}")
    if encrypted_members:
        failures.append(f"encrypted members: {encrypted_members}")
    if symlink_members:
        failures.append(f"symbolic-link members: {symlink_members}")
    if frozen_markers:
        failures.append(f"frozen-video markers: {frozen_markers}")
    if unsupported_top_level:
        failures.append(f"members outside M3OT root: {unsupported_top_level}")
    if coco_invalid_images:
        failures.append(f"invalid train COCO images: {coco_invalid_images}")
    if coco_invalid_annotations:
        failures.append(f"invalid train COCO annotations: {coco_invalid_annotations}")
    if coco_missing_image_members:
        failures.append(f"train COCO images missing from ZIP: {coco_missing_image_members}")
    if coco_duplicate_image_members:
        failures.append(f"duplicate train COCO image mappings: {coco_duplicate_image_members}")
    if coco_image_count < minimum_rgb_train_images:
        failures.append(f"minimum RGB train images: {coco_image_count} < {minimum_rgb_train_images}")
    if coco_annotation_count < minimum_rgb_train_annotations:
        failures.append(f"minimum RGB train annotations: {coco_annotation_count} < {minimum_rgb_train_annotations}")
    if mapped_coco_members != {
        name for name in normalized_names if "/rgb/train/" in name and name.lower().endswith(".png")
    }:
        failures.append("RGB train central-directory images do not exactly match COCO image mappings")

    return {
        "schema_version": "m3ot-archive-audit-v1",
        "status": "pass" if not failures else "fail",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": actual_md5,
        "archive_sha256": actual_sha256,
        "official_bytes": expected_bytes,
        "official_md5": expected_md5.lower(),
        "source_evidence": str(source_evidence),
        "source_evidence_sha256": sha256_file(source_evidence),
        "license": license_name,
        "zip_member_count": len(normalized_names),
        "zip_uncompressed_bytes": sum(member_bytes.values()),
        "image_member_counts": dict(sorted(member_counts.items())),
        "image_member_bytes": dict(sorted(member_bytes.items())),
        "rgb_train_sequences": {key: sorted(value) for key, value in train_sequences.items()},
        "rgb_train_coco_images": coco_image_count,
        "rgb_train_coco_annotations": coco_annotation_count,
        "rgb_train_small_annotations_lt_1024": coco_small_annotation_count,
        "rgb_train_small_fraction": coco_small_annotation_count / coco_annotation_count if coco_annotation_count else 0.0,
        "rgb_train_unique_track_keys": len(coco_track_keys),
        "official_class_taxonomy": ["vehicle"],
        "fine_body_labels_available": False,
        "color_labels_available": False,
        "duplicate_normalized_names": duplicate_names,
        "unsafe_member_paths": unsafe_paths,
        "encrypted_members": encrypted_members,
        "symlink_members": symlink_members,
        "frozen_markers": frozen_markers,
        "opened_zip_members": opened_members,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "ir_payload_opened": False,
        "images_extracted": False,
        "training_started": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "admission": "RGB train only; vehicle boxes and track IDs may support small-target/domain consistency. Fine body and color truth remain unknown unless a later conservative multi-frame teacher audit accepts them.",
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed M3OT official ZIP and RGB-train COCO audit.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--minimum-rgb-train-images", type=int, default=8000)
    parser.add_argument("--minimum-rgb-train-annotations", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_archive(
        args.archive.resolve(),
        args.source_evidence.resolve(),
        expected_bytes=args.expected_bytes,
        expected_md5=args.expected_md5,
        minimum_rgb_train_images=args.minimum_rgb_train_images,
        minimum_rgb_train_annotations=args.minimum_rgb_train_annotations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
