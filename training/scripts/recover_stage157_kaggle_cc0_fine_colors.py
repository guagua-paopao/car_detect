#!/usr/bin/env python3
"""Recover fine color truth from the original CC0 Pascal-VOC annotations.

The legacy Kaggle supplement merged gray/silver, yellow/orange and
brown/beige.  This tool replays the original crop operation and requires the
encoded JPEG SHA256 to match exactly before accepting a fine label.  A row
that cannot be matched uniquely is failed closed to unknown.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image


FINE_COLOR_MAP = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "brown": "brown",
    # The v2 taxonomy has no dedicated orange or beige class.  Mapping them
    # to yellow/brown would fabricate fine truth, so retain them as `other`.
    "orange": "other",
    "beige": "other",
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_raw_label(value: str) -> tuple[str, str]:
    parts = value.strip().lower().split("_", 1)
    if len(parts) != 2:
        return "unknown", "unknown"
    return FINE_COLOR_MAP.get(parts[0], "unknown"), parts[1]


def encoded_crop_sha(image: Image.Image, node: ET.Element) -> str | None:
    box = node.find("bndbox")
    if box is None:
        return None
    try:
        left = max(0, math.floor(float(box.findtext("xmin", "nan"))))
        top = max(0, math.floor(float(box.findtext("ymin", "nan"))))
        right = min(image.width, math.ceil(float(box.findtext("xmax", "nan"))))
        bottom = min(image.height, math.ceil(float(box.findtext("ymax", "nan"))))
    except (TypeError, ValueError, OverflowError):
        return None
    if right <= left or bottom <= top:
        return None
    buffer = io.BytesIO()
    image.crop((left, top, right, bottom)).save(
        buffer, format="JPEG", quality=95, optimize=True
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def recover_rows(rows: list[dict[str, str]], archive_path: Path) -> tuple[list[dict[str, str]], Counter]:
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row.get("source_frame_id", "")].append(index)
    counters: Counter = Counter()
    output = [dict(row) for row in rows]
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        for image_name, indexes in sorted(by_image.items()):
            xml_name = str(PurePosixPath(image_name).with_suffix(".xml"))
            if not image_name or image_name not in archive_names or xml_name not in archive_names:
                for index in indexes:
                    output[index].update({
                        "color_coarse_source": output[index].get("color", "unknown"),
                        "color": "unknown",
                        "color_supervised": "false",
                        "fine_color_recovery": "missing_source_member",
                    })
                    counters["missing_source_member"] += 1
                continue
            try:
                image = Image.open(io.BytesIO(archive.read(image_name))).convert("RGB")
                root = ET.fromstring(archive.read(xml_name))
            except Exception:
                for index in indexes:
                    output[index].update({
                        "color_coarse_source": output[index].get("color", "unknown"),
                        "color": "unknown",
                        "color_supervised": "false",
                        "fine_color_recovery": "unreadable_source_member",
                    })
                    counters["unreadable_source_member"] += 1
                continue
            candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for node in root.findall("object"):
                digest = encoded_crop_sha(image, node)
                if not digest:
                    continue
                raw_label = node.findtext("name", "").strip().lower()
                fine_color, _ = parse_raw_label(raw_label)
                candidates[digest].append((raw_label, fine_color))
            for index in indexes:
                row = output[index]
                coarse = row.get("color", "unknown")
                digest = (row.get("crop_sha256") or row.get("sha256") or "").lower()
                matches = candidates.get(digest, [])
                row["color_coarse_source"] = coarse
                if len(matches) != 1 or matches[0][1] == "unknown":
                    row.update({
                        "source_raw_color_label": matches[0][0] if len(matches) == 1 else "",
                        "color": "unknown",
                        "color_supervised": "false",
                        "fine_color_recovery": "ambiguous_or_unmatched",
                    })
                    counters["ambiguous_or_unmatched"] += 1
                    continue
                raw_label, fine_color = matches[0]
                row.update({
                    "source_raw_color_label": raw_label,
                    "color": fine_color,
                    "color_supervised": "true",
                    "fine_color_recovery": "exact_encoded_crop_sha256",
                })
                counters["exact_recovered"] += 1
                counters[f"fine_color_{fine_color}"] += 1
    return output, counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.archive, args.input_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    archive_sha = sha256_file(args.archive)
    input_sha = sha256_file(args.input_manifest)
    if archive_sha != args.expected_archive_sha256.lower():
        raise RuntimeError("archive SHA256 mismatch")
    if input_sha != args.expected_input_sha256.lower():
        raise RuntimeError("input manifest SHA256 mismatch")
    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty input manifest")
    for row in rows:
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found")
        if row.get("split") != "train":
            raise RuntimeError("legacy CC0 manifest must be train-only")
        if row.get("source_license") != "CC0-1.0":
            raise RuntimeError("unexpected source license")
    recovered, counters = recover_rows(rows, args.archive)
    recovered_count = counters["exact_recovered"]
    failures = []
    if recovered_count != len(rows):
        failures.append(f"fine labels recovered for {recovered_count}/{len(rows)} rows")
    fields: list[str] = []
    for row in recovered:
        for field in row:
            if field not in fields:
                fields.append(field)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(recovered)
    report = {
        "schema_version": "stage157-kaggle-cc0-fine-color-recovery-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail_closed",
        "inputs": {
            "archive": str(args.archive.resolve()),
            "archive_sha256": archive_sha,
            "manifest": str(args.input_manifest.resolve()),
            "manifest_sha256": input_sha,
            "rows": len(rows),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256_file(args.output_manifest),
            "rows": len(recovered),
            "fine_color_counts": dict(sorted(Counter(
                row["color"] for row in recovered if row.get("color_supervised") == "true"
            ).items())),
        },
        "counters": dict(sorted(counters.items())),
        "policy": {
            "exact_encoded_crop_sha256_required": True,
            "merged_labels_split_by_inference": False,
            "orange_mapped_to": "other",
            "beige_mapped_to": "other",
            "unmatched_rows": "unknown_unsupervised",
            "test_rows_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "counts": report["output"]["fine_color_counts"], "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
