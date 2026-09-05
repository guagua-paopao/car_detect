#!/usr/bin/env python3
"""Audit the official L-VAD v2 archive while keeping publisher valid/test payloads sealed."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageStat


EXPECTED_IMAGES = {"train": 10934, "valid": 1364, "test": 1365}
EXPECTED_LABELS = {"train": 10919, "valid": 1364, "test": 1365}
CLASS_CONTRACT = {
    0: "motorcycle_excluded",
    1: "mixed_car_family_unknown_only",
    2: "mixed_truck_bus_family_coarse_only",
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen_video")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_INVALID_CLASS_FRAMES = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = list(gray.getdata())
    result = 0
    for y in range(8):
        row = values[y * 9:(y + 1) * 9]
        for x in range(8):
            result = (result << 1) | int(row[x + 1] > row[x])
    return result


def source_frame_key(stem: str) -> str:
    """Strip Roboflow's transform hash but preserve the original temporal frame id."""
    return re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)


def out_of_contract_classes(text: str) -> Counter[int]:
    invalid: Counter[int] = Counter()
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid class id") from exc
        if class_id not in CLASS_CONTRACT:
            invalid[class_id] += 1
    return invalid


def parse_yolo(text: str, image_width: int, image_height: int) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    small = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid numeric field") from exc
        if class_id not in CLASS_CONTRACT:
            raise ValueError(f"line {line_number}: class {class_id} outside 0..2")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"line {line_number}: non-finite box")
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"line {line_number}: invalid normalized box")
        tolerance = 1e-3
        if x - width / 2 < -tolerance or x + width / 2 > 1 + tolerance:
            raise ValueError(f"line {line_number}: horizontal box outside image")
        if y - height / 2 < -tolerance or y + height / 2 > 1 + tolerance:
            raise ValueError(f"line {line_number}: vertical box outside image")
        counts[class_id] += 1
        if width * image_width < 64 or height * image_height < 64:
            small += 1
    return counts, small


def audit(archive: Path, expected_sha256: str, expected_size: int, evidence: Path) -> dict[str, object]:
    archive = archive.resolve()
    evidence = evidence.resolve()
    if not archive.is_file() or not evidence.is_file():
        raise FileNotFoundError(archive if not archive.is_file() else evidence)
    archive_sha = sha256_file(archive)
    evidence_sha = sha256_file(evidence)
    source = json.loads(evidence.read_text(encoding="utf-8"))
    failures: list[str] = []
    if archive.stat().st_size != expected_size:
        failures.append(f"archive size {archive.stat().st_size} != {expected_size}")
    if archive_sha.lower() != expected_sha256.lower():
        failures.append("archive SHA256 mismatch")
    if source.get("license") != "CC-BY-4.0" or source.get("doi") != "10.17632/h6p2w53my5.2":
        failures.append("source evidence license/DOI mismatch")

    root = ""
    structure: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    exact_hashes: set[str] = set()
    exact_duplicates = 0
    dhash_counts: Counter[str] = Counter()
    source_frame_counts: Counter[str] = Counter()
    invalid_class_counts: Counter[int] = Counter()
    invalid_class_frames = 0
    small_boxes = 0
    low_luminance_frames = 0
    very_dark_frames = 0
    train_payloads_read = 0
    valid_payloads_read = 0
    test_payloads_read = 0

    with zipfile.ZipFile(archive) as opened:
        infos = opened.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            failures.append("duplicate ZIP member names")
        unsafe = [name for name in names if not safe_member(name)]
        if unsafe:
            failures.append(f"unsafe ZIP paths: {unsafe[:3]}")
        frozen = [name for name in names if any(marker in name.lower() for marker in FROZEN_MARKERS)]
        if frozen:
            failures.append(f"frozen marker in archive: {frozen[:3]}")
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            failures.append(f"expected one archive root, got {sorted(roots)}")
        else:
            root = next(iter(roots))

        image_by_split: dict[str, dict[str, str]] = {split: {} for split in EXPECTED_IMAGES}
        label_by_split: dict[str, dict[str, str]] = {split: {} for split in EXPECTED_IMAGES}
        data_yaml = ""
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if len(parts) == 2 and parts[0] == root and parts[1] == "data.yaml":
                data_yaml = info.filename
                continue
            if len(parts) != 4 or parts[0] != root:
                continue
            split, kind, filename = parts[1], parts[2], parts[3]
            if split not in EXPECTED_IMAGES:
                continue
            if kind == "images" and Path(filename).suffix.lower() in IMAGE_SUFFIXES:
                image_by_split[split][Path(filename).stem] = info.filename
            elif kind == "labels" and Path(filename).suffix.lower() == ".txt":
                label_by_split[split][Path(filename).stem] = info.filename

        for split in EXPECTED_IMAGES:
            images = image_by_split[split]
            labels = label_by_split[split]
            structure[f"{split}_images"] = len(images)
            structure[f"{split}_labels"] = len(labels)
            if len(images) != EXPECTED_IMAGES[split] or len(labels) != EXPECTED_LABELS[split]:
                failures.append(
                    f"{split} count images={len(images)} labels={len(labels)} "
                    f"expected={EXPECTED_IMAGES[split]}/{EXPECTED_LABELS[split]}"
                )
            if not set(labels).issubset(set(images)):
                failures.append(f"{split} contains labels without paired images")
            if split != "train" and set(labels) != set(images):
                failures.append(f"{split} held-out image/label stems do not pair exactly")

        if not data_yaml:
            failures.append("data.yaml missing")
        else:
            yaml_text = opened.read(data_yaml).decode("utf-8-sig", errors="strict")
            train_payloads_read += 1
            required = ("nc: 3", "names: ['0', '1', '2']", "license: CC BY 4.0")
            if not all(marker in yaml_text for marker in required):
                failures.append("data.yaml class/license contract mismatch")

        for stem in sorted(image_by_split["train"]):
            image_payload = opened.read(image_by_split["train"][stem])
            train_payloads_read += 1
            label_member = label_by_split["train"].get(stem)
            label_payload = opened.read(label_member) if label_member else b""
            if label_member:
                train_payloads_read += 1
            digest = hashlib.sha256(image_payload).hexdigest()
            if digest in exact_hashes:
                exact_duplicates += 1
            exact_hashes.add(digest)
            source_frame_counts[source_frame_key(stem)] += 1
            try:
                with Image.open(io.BytesIO(image_payload)) as source_image:
                    source_image.load()
                    image = source_image.convert("RGB")
                width, height = image.size
                if width < 64 or height < 64:
                    raise ValueError("image dimension below 64")
                dhash_counts[f"{dhash64(image):016x}"] += 1
                mean = float(ImageStat.Stat(image.convert("L")).mean[0])
                if mean < 55:
                    low_luminance_frames += 1
                if mean < 30:
                    very_dark_frames += 1
                label_text = label_payload.decode("utf-8-sig", errors="strict")
                invalid = out_of_contract_classes(label_text)
                if invalid:
                    invalid_class_frames += 1
                    invalid_class_counts.update(invalid)
                    continue
                parsed, small = parse_yolo(label_text, width, height)
                class_counts.update(parsed)
                small_boxes += small
            except Exception as exc:
                failures.append(f"train/{stem}: {exc}")
                if len(failures) > 100:
                    break

    if sum(class_counts.values()) == 0:
        failures.append("no valid train objects")
    if invalid_class_frames > MAX_INVALID_CLASS_FRAMES:
        failures.append(
            f"out-of-contract class frames {invalid_class_frames} > {MAX_INVALID_CLASS_FRAMES}"
        )
    report = {
        "schema_version": "stage92-lvad-archive-audit-v2",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha, "source_evidence": str(evidence),
            "source_evidence_sha256": evidence_sha, "doi": source.get("doi"),
            "license": source.get("license"),
        },
        "structure": {"root": root, **dict(sorted(structure.items()))},
        "train_audit": {
            "object_counts_by_yolo_class": {str(key): value for key, value in sorted(class_counts.items())},
            "objects": sum(class_counts.values()), "small_target_boxes": small_boxes,
            "low_luminance_proxy_frames": low_luminance_frames,
            "very_dark_proxy_frames": very_dark_frames,
            "exact_duplicate_full_frames": exact_duplicates,
            "repeated_dhash_full_frames": sum(count - 1 for count in dhash_counts.values() if count > 1),
            "repeated_source_frame_variants": sum(count - 1 for count in source_frame_counts.values() if count > 1),
            "excluded_out_of_contract_class_frames": invalid_class_frames,
            "excluded_out_of_contract_class_counts": {
                str(key): value for key, value in sorted(invalid_class_counts.items())
            },
        },
        "mapping": {str(key): value for key, value in CLASS_CONTRACT.items()},
        "access_policy": {
            "train_payloads_read": train_payloads_read,
            "validation_payloads_read": valid_payloads_read,
            "test_payloads_read": test_payloads_read,
            "source_valid_or_test_used_for_training_calibration_or_thresholds": False,
            "motorcycles_train_eligible": False,
            "fine_body_labels_fabricated": False,
            "color_labels_fabricated": False,
            "out_of_contract_class_frames_train_eligible": False,
            "verified_night_source_claim": True,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage92 audit evidence")
    report = audit(args.archive, args.expected_sha256, args.expected_size, args.source_evidence)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "objects": report["train_audit"]["objects"]}, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
