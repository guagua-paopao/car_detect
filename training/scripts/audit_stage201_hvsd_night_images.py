#!/usr/bin/env python3
"""Audit the small CC-BY HVSD NIGHT archive and isolate the unprocessed night quadrant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


EXPECTED_BYTES = 4235324
EXPECTED_SHA256 = "3cc2b812220d765998e5976b9abeb9c05ab5101337a625f0b74f81994f8773ba"
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if any(marker in str(value).lower() for value in (args.archive, args.output_root) for marker in FROZEN_MARKERS):
        raise ValueError("frozen-video path is forbidden")
    if args.archive.stat().st_size != EXPECTED_BYTES or sha256_file(args.archive) != EXPECTED_SHA256:
        raise RuntimeError("archive integrity mismatch")

    args.output_root.mkdir(parents=True)
    frames_root = args.output_root / "frames"
    frames_root.mkdir()
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP CRC failure")
        members = []
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                raise RuntimeError(f"unsafe member: {info.filename}")
            if not info.is_dir() and pure.suffix.lower() == ".png":
                members.append(info)
        if len(members) != 10:
            raise RuntimeError(f"unexpected PNG count: {len(members)}")
        for index, info in enumerate(sorted(members, key=lambda item: item.filename)):
            payload = archive.read(info)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"decode failure: {info.filename}")
            height, width = image.shape[:2]
            if (width, height) != (640, 480):
                raise RuntimeError(f"unexpected composite geometry: {info.filename} {width}x{height}")
            # Each source PNG is a 2x2 experiment montage. Only the upper-left
            # quadrant is the original frame; remove its 18-pixel title strip.
            image = image[32 : height // 2, : width // 2].copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output_path = frames_root / PurePosixPath(info.filename).name
            encoded_ok, encoded = cv2.imencode(".png", image)
            if not encoded_ok:
                raise RuntimeError(f"encode failure: {info.filename}")
            isolated_payload = encoded.tobytes()
            output_path.write_bytes(isolated_payload)
            rows.append({
                "split": "train", "source_dataset": "HVSD-v1", "source_doi": "10.17632/zdgmdg7p25.1",
                "source_license": "CC BY 4.0", "source_member": info.filename,
                "image_path": str(output_path), "source_group": "hvsd-night-experiment-sequence-1",
                "frame_order": index, "confirmed_natural_night": "true", "scene_origin": "source_explicit_night_archive",
                "body_type": "unknown", "body_type_supervised": "false", "color": "unknown", "color_supervised": "false",
                "width": image.shape[1], "height": image.shape[0], "pixel_sha256": sha256_bytes(image.tobytes()),
                "file_sha256": sha256_bytes(isolated_payload), "dhash64": f"{dhash64(gray):016x}",
                "mean_luma": round(float(gray.mean()), 6), "median_luma": round(float(np.median(gray)), 6),
                "dark_fraction": round(float((gray <= 72).mean()), 6),
                "effective_representative": str(index in {0, 2, 4, 6, 9}).lower(),
                "train_role": "unlabeled_night_domain_consistency_only",
            })

    if len({row["pixel_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("exact duplicate frame detected")
    manifest = args.output_root / "stage201-hvsd-night-manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    columns, tile_w, tile_h = 5, 330, 235
    sheet = Image.new("RGB", (columns * tile_w, 2 * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(row["image_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 30))
        column, sheet_row = index % columns, index // columns
        x0, y0 = column * tile_w, sheet_row * tile_h
        sheet.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 26 + (tile_h - 28 - image.height) // 2))
        draw.text((x0 + 4, y0 + 4), f"{Path(row['source_member']).name} mean={row['mean_luma']} dark={row['dark_fraction']}", fill="black")
    contact = args.output_root / "stage201-hvsd-night-contact-sheet.jpg"
    sheet.save(contact, quality=94)

    report = {
        "schema_version": "stage201-hvsd-night-audit-v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_component_unlabeled_night_domain_only_no_attribute_training",
        "source": {"doi": "10.17632/zdgmdg7p25.1", "license": "CC BY 4.0", "archive_bytes": EXPECTED_BYTES, "archive_sha256": EXPECTED_SHA256},
        "counts": {"decoded_composite_frames": len(rows), "isolated_original_quadrants": len(rows), "processed_quadrants_excluded": 30, "exact_duplicate_frames": 0, "effective_sequence_cap5": 5, "body_truth": 0, "color_truth": 0},
        "outputs": {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "contact_sheet": str(contact), "contact_sheet_sha256": sha256_file(contact), "frames_root": str(frames_root)},
        "policy": {"role": "train-only unlabeled natural-night consistency", "original_quadrant_only": True, "title_strip_removed_pixels": 32, "test_accessed": False, "frozen_video_used": False, "production_model_modified": False, "deployment_performed": False},
    }
    report_path = args.output_root / "stage201-hvsd-night-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
