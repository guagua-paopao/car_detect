#!/usr/bin/env python3
"""Materialize only ExDark's official vehicle training split.

This tool deliberately does not extract or decode validation/test images.  It
uses the official imageclasslist.txt as the sole split authority and audits tar
members before writing selected training images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image


EXPECTED = {
    "car": {
        "class_id": 5,
        "count": 250,
        "archive": "car.tar.gz",
        "md5": "c4c8e6ced85ac27fff16dc5560461f21",
        "sha256": "8f1554f3724468179e4f12d2116aaa2a5fce257f1bb3968d5049bf10430d92e5",
    },
    "bus": {
        "class_id": 4,
        "count": 250,
        "archive": "bus.tar.gz",
        "md5": "a8a7363be3282873e3c31418834c9560",
        "sha256": "064c12fe382d26062c4370eaa6af322e7adcf40ae2d0584047f63fbaf3851df5",
    },
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_hex(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
    return f"{value:016x}"


def parse_split(path: Path) -> tuple[list[dict], dict[str, int]]:
    selected: list[dict] = []
    all_names: set[str] = set()
    split_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip()
        if "Train/Val/Test" not in header:
            raise ValueError(f"unexpected split header: {header!r}")
        for lineno, raw in enumerate(handle, start=2):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"bad split row at line {lineno}: {raw!r}")
            name, class_id, light_id, inout_id, split_id = parts
            if name in all_names:
                raise ValueError(f"duplicate official image name: {name}")
            all_names.add(name)
            split_counts[split_id] += 1
            if int(class_id) in (4, 5) and int(split_id) == 1:
                selected.append(
                    {
                        "image_name": name,
                        "official_class_id": int(class_id),
                        "lighting_id": int(light_id),
                        "indoor_outdoor_id": int(inout_id),
                        "official_split_id": int(split_id),
                    }
                )
    if len(all_names) != 7363:
        raise ValueError(f"expected 7363 unique split names, found {len(all_names)}")
    return selected, dict(sorted(split_counts.items()))


def audit_members(tf: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    by_basename: dict[str, tarfile.TarInfo] = {}
    for member in tf.getmembers():
        posix = PurePosixPath(member.name)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"link/device archive member forbidden: {member.name}")
        if not member.isfile():
            continue
        basename = posix.name
        if basename.startswith("._"):
            continue
        if basename in by_basename:
            raise ValueError(f"duplicate regular-file basename in archive: {basename}")
        by_basename[basename] = member
    return by_basename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    images_root = args.output_root / "images"
    images_root.mkdir()

    split_path = args.official_repo / "Groundtruth" / "imageclasslist.txt"
    selected, split_counts = parse_split(split_path)
    selected_by_class: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        selected_by_class[row["official_class_id"]].append(row)

    manifest: list[dict] = []
    archive_evidence: dict[str, dict] = {}
    for class_name, spec in EXPECTED.items():
        rows = selected_by_class[spec["class_id"]]
        if len(rows) != spec["count"]:
            raise ValueError(
                f"expected {spec['count']} official train rows for {class_name}, found {len(rows)}"
            )
        archive = args.source_root / spec["archive"]
        actual_md5 = digest(archive, "md5")
        actual_sha = digest(archive, "sha256")
        if actual_md5 != spec["md5"] or actual_sha != spec["sha256"]:
            raise ValueError(f"archive digest mismatch: {archive}")
        output_class = images_root / class_name
        output_class.mkdir()
        with tarfile.open(archive, "r:gz") as tf:
            members = audit_members(tf)
            wanted = {row["image_name"] for row in rows}
            missing = sorted(wanted - set(members))
            if missing:
                raise ValueError(f"missing {class_name} train files: {missing[:10]}")
            for row in sorted(rows, key=lambda item: item["image_name"]):
                member = members[row["image_name"]]
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read tar member: {member.name}")
                payload = extracted.read()
                pixel_sha = hashlib.sha256(payload).hexdigest()
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    dhash = dhash_hex(image)
                if width < 16 or height < 16:
                    raise ValueError(f"implausibly small image: {row['image_name']} {width}x{height}")
                destination = output_class / row["image_name"]
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                with destination.open("xb") as handle:
                    handle.write(payload)
                manifest.append(
                    {
                        **row,
                        "official_class_name": class_name,
                        "relative_path": destination.relative_to(args.output_root).as_posix(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "bytes": len(payload),
                        "sha256": pixel_sha,
                        "dhash64": dhash,
                    }
                )
        archive_evidence[class_name] = {
            "path": str(archive),
            "md5": actual_md5,
            "sha256": actual_sha,
            "selected_train_images": len(rows),
        }

    sha_groups: dict[str, list[str]] = defaultdict(list)
    dhash_groups: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        sha_groups[row["sha256"]].append(row["image_name"])
        dhash_groups[row["dhash64"]].append(row["image_name"])
    exact_duplicate_groups = [names for names in sha_groups.values() if len(names) > 1]
    identical_dhash_groups = [names for names in dhash_groups.values() if len(names) > 1]

    manifest_path = args.output_root / "train_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    split_sha = digest(split_path, "sha256")
    report = {
        "stage": "stage226-exdark-train-only-materialization-r1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_train_pixels_only",
        "official_repository_commit": (
            args.official_repo.parent / "official-repository.commit"
        ).read_text(encoding="utf-8").strip(),
        "official_split_file": str(split_path),
        "official_split_file_sha256": split_sha,
        "official_full_split_counts": split_counts,
        "selected_train_images": len(manifest),
        "selected_counts": dict(Counter(row["official_class_name"] for row in manifest)),
        "lighting_id_counts": dict(
            sorted(Counter(str(row["lighting_id"]) for row in manifest).items())
        ),
        "indoor_outdoor_id_counts": dict(
            sorted(Counter(str(row["indoor_outdoor_id"]) for row in manifest).items())
        ),
        "archives": archive_evidence,
        "decoded_images": len(manifest),
        "unreadable_images": 0,
        "exact_duplicate_groups_within_selected_train": exact_duplicate_groups,
        "identical_dhash_groups_within_selected_train": identical_dhash_groups,
        "validation_pixels_opened": 0,
        "test_pixels_opened": 0,
        "frozen_video_used": False,
        "native_vehicle_color_labels": False,
        "admission_state": "not_yet_supervised; detector crop and fail-closed color audit required",
        "production_model_modified": False,
        "training_started": False,
        "deployment_performed": False,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums_path = args.output_root / "SHA256SUMS"
    with sums_path.open("w", encoding="utf-8") as handle:
        for path in (manifest_path, report_path, split_path):
            handle.write(f"{digest(path, 'sha256')}  {path}\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
