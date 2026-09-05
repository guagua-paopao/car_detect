#!/usr/bin/env python3
"""Audit the official MY-VID v2 ZIP without decoding any images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_BYTES = 2637612140
EXPECTED_MD5 = "cd506cf6e6ba8ef41d030b08bcc9d006"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalized_member(name: str) -> str:
    return str(PurePosixPath(name.replace("\\", "/")))


def member_split(name: str) -> tuple[str, str] | None:
    normalized = "/" + normalized_member(name).lower().lstrip("/")
    for split, aliases in {
        "train": ("train",),
        "validation": ("valid", "val", "validation"),
        "test": ("test",),
    }.items():
        for alias in aliases:
            marker = f"/{alias}/"
            if marker in normalized:
                part = normalized.split(marker, 1)[1]
                if part.startswith("images/"):
                    return split, "image"
                if part.startswith("labels/") and normalized.endswith(".txt"):
                    return split, "label"
    return None


def audit(archive: Path) -> dict[str, object]:
    if any(marker in str(archive).lower() for marker in FROZEN_MARKERS):
        raise ValueError("frozen-video path is forbidden")
    failures: list[str] = []
    size = archive.stat().st_size
    if size != EXPECTED_BYTES:
        failures.append(f"archive size mismatch: {size}")
    md5 = digest(archive, "md5")
    sha256 = digest(archive, "sha256")
    if md5 != EXPECTED_MD5:
        failures.append(f"archive MD5 mismatch: {md5}")

    counts: Counter[tuple[str, str]] = Counter()
    normalized_counts: Counter[str] = Counter()
    unsafe: list[str] = []
    yaml_members: list[str] = []
    top_levels: set[str] = set()
    with zipfile.ZipFile(archive) as package:
        bad_member = package.testzip()
        if bad_member:
            failures.append(f"CRC failure: {bad_member}")
        members = package.infolist()
        for member in members:
            raw = member.filename.replace("\\", "/")
            pure = PurePosixPath(raw)
            if pure.is_absolute() or ".." in pure.parts:
                unsafe.append(member.filename)
                continue
            normalized = normalized_member(raw)
            normalized_counts[normalized] += 1
            if pure.parts:
                top_levels.add(pure.parts[0])
            if pure.name.lower() == "data.yaml":
                yaml_members.append(member.filename)
            split_kind = member_split(raw)
            if split_kind and pure.suffix.lower() in IMAGE_SUFFIXES | {".txt"}:
                counts[split_kind] += 1
        yaml_texts = {
            name: package.read(name).decode("utf-8", errors="replace")
            for name in yaml_members
        }

    duplicate_names = sum(value - 1 for value in normalized_counts.values() if value > 1)
    if unsafe:
        failures.append(f"unsafe archive paths: {len(unsafe)}")
    if duplicate_names:
        failures.append(f"duplicate normalized member names: {duplicate_names}")
    if len(yaml_members) != 1:
        failures.append(f"expected one data.yaml, found {len(yaml_members)}")
    if counts[("train", "image")] <= 0 or counts[("train", "label")] <= 0:
        failures.append("train images or labels missing")

    return {
        "stage": "stage186_myvid_v2_archive_audit_r1",
        "status": "pass" if not failures else "fail_closed",
        "source": {
            "name": "MY-VID v2",
            "doi": "10.5281/zenodo.17861464",
            "license": "CC BY 4.0",
            "expected_bytes": EXPECTED_BYTES,
            "expected_md5": EXPECTED_MD5,
            "actual_bytes": size,
            "actual_md5": md5,
            "sha256": sha256,
        },
        "archive": {
            "members": sum(normalized_counts.values()),
            "top_levels": sorted(top_levels),
            "unsafe_paths": len(unsafe),
            "duplicate_normalized_names": duplicate_names,
            "yaml_members": yaml_members,
            "yaml_text": yaml_texts,
            "split_counts": {
                split: {
                    "images": counts[(split, "image")],
                    "labels": counts[(split, "label")],
                }
                for split in ("train", "validation", "test")
            },
        },
        "scope": {
            "image_pixels_decoded": 0,
            "validation_images_opened": 0,
            "test_images_opened": 0,
            "frozen_video_used": False,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
