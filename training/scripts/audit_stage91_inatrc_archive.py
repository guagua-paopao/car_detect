#!/usr/bin/env python3
"""Audit InaTRC v1 without opening validation or test payloads."""

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

from PIL import Image, ImageStat


EXPECTED_SPLIT_IMAGES = {"train": 2600, "val": 325, "test": 325}
EXPECTED_CLASSES_TEXT = ["1", "2", "3", "4", "5"]
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen_video")


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


def parse_yolo(text: str, image_width: int, image_height: int) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    small = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid numeric field") from exc
        if class_id not in range(5):
            raise ValueError(f"line {line_number}: class {class_id} outside 0..4")
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
    failures: list[str] = []
    if archive.stat().st_size != expected_size:
        failures.append(f"archive size {archive.stat().st_size} != {expected_size}")
    if archive_sha.lower() != expected_sha256.lower():
        failures.append("archive SHA256 mismatch")
    source = json.loads(evidence.read_text(encoding="utf-8"))
    if source.get("license") != "CC BY 4.0" or source.get("doi") != "10.17632/kddgphck3b.1":
        failures.append("source evidence license/DOI mismatch")

    counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    exact_hashes: set[str] = set()
    exact_duplicates = 0
    dhash_counts: Counter[str] = Counter()
    small_boxes = 0
    low_luminance_frames = 0
    train_payloads_read = 0
    validation_payloads_read = 0
    test_payloads_read = 0
    root = ""

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

        image_by_split: dict[str, dict[str, str]] = {split: {} for split in EXPECTED_SPLIT_IMAGES}
        label_by_split: dict[str, dict[str, str]] = {split: {} for split in EXPECTED_SPLIT_IMAGES}
        classes_member = ""
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 4 or parts[0] != root:
                if len(parts) == 2 and parts[0] == root and parts[1] == "README.md":
                    counts["readme_members"] += 1
                continue
            split, kind, filename = parts[1], parts[2], parts[3]
            if split not in EXPECTED_SPLIT_IMAGES:
                continue
            if kind == "images" and Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_by_split[split][Path(filename).stem] = info.filename
            elif kind == "labels" and filename.lower() == "classes.txt":
                if split == "train":
                    classes_member = info.filename
            elif kind == "labels" and Path(filename).suffix.lower() == ".txt":
                label_by_split[split][Path(filename).stem] = info.filename

        for split, expected in EXPECTED_SPLIT_IMAGES.items():
            counts[f"{split}_images"] = len(image_by_split[split])
            counts[f"{split}_labels"] = len(label_by_split[split])
            if len(image_by_split[split]) != expected or len(label_by_split[split]) != expected:
                failures.append(
                    f"{split} pair count images={len(image_by_split[split])} labels={len(label_by_split[split])} expected={expected}"
                )
            if set(image_by_split[split]) != set(label_by_split[split]):
                failures.append(f"{split} image/label stems do not pair exactly")

        if not classes_member:
            failures.append("train classes.txt missing")
        else:
            classes = opened.read(classes_member).decode("utf-8-sig", errors="strict").split()
            train_payloads_read += 1
            if classes != EXPECTED_CLASSES_TEXT:
                failures.append(f"classes.txt mismatch: {classes}")

        readme_member = f"{root}/README.md"
        try:
            readme = opened.read(readme_member).decode("utf-8-sig", errors="replace")
            train_payloads_read += 1
            if "Kelas-1" not in readme or "Kelas-5" not in readme or "YOLO" not in readme:
                failures.append("README class/format contract missing")
        except KeyError:
            failures.append("README.md missing")

        for stem in sorted(image_by_split["train"]):
            image_payload = opened.read(image_by_split["train"][stem])
            label_payload = opened.read(label_by_split["train"][stem])
            train_payloads_read += 2
            digest = hashlib.sha256(image_payload).hexdigest()
            if digest in exact_hashes:
                exact_duplicates += 1
            exact_hashes.add(digest)
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
                parsed, small = parse_yolo(label_payload.decode("utf-8-sig", errors="strict"), width, height)
                class_counts.update(parsed)
                small_boxes += small
            except Exception as exc:
                failures.append(f"train/{stem}: {exc}")
                if len(failures) > 100:
                    break

    if sum(class_counts.values()) == 0:
        failures.append("no valid train objects")
    report = {
        "schema_version": "stage91-inatrc-archive-audit-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha, "source_evidence": str(evidence),
            "source_evidence_sha256": evidence_sha, "doi": source.get("doi"),
            "license": source.get("license"), "attribution": source.get("attribution"),
        },
        "structure": {"root": root, **dict(sorted(counts.items()))},
        "train_audit": {
            "object_counts_by_yolo_class": {str(key): value for key, value in sorted(class_counts.items())},
            "objects": sum(class_counts.values()), "small_target_boxes": small_boxes,
            "low_luminance_proxy_frames": low_luminance_frames,
            "exact_duplicate_full_frames": exact_duplicates,
            "repeated_dhash_full_frames": sum(count - 1 for count in dhash_counts.values() if count > 1),
        },
        "access_policy": {
            "train_payloads_read": train_payloads_read,
            "validation_payloads_read": validation_payloads_read,
            "test_payloads_read": test_payloads_read,
            "validation_or_test_used_for_training_calibration_or_thresholds": False,
            "class_0_fine_type_fabricated": False,
            "color_labels_fabricated": False,
            "verified_night_truth_claimed": False,
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
        raise FileExistsError("refusing to overwrite Stage91 audit evidence")
    report = audit(args.archive, args.expected_sha256, args.expected_size, args.source_evidence)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "objects": report["train_audit"]["objects"]}, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
