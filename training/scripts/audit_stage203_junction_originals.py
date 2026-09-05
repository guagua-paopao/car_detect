#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


DATASET_ID = "vwjg6b7kpt"
DATASET_VERSION = 1
FOLDER_ID = "7856d16f-d1b9-4677-9e2d-377d3b0ecc62"
PUBLIC_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
EXPECTED_LABEL_NAME = "sample_labels.csv"
EXPECTED_LABEL_BYTES = 730411
EXPECTED_LABEL_SHA256 = "fef3a4d0963aa80fa0ce931f8f730484d366f72a890c7b3a46f85ed1046a12a1"
EXPECTED_ORIGINAL_IMAGES = 159
EXPECTED_ORIGINAL_BYTES = 24830318
EXACT_BODY_MAP = {
    "bus": "bus",
    "van": "van",
    "light_truck": "light_truck",
    "heavy_truck": "heavy_truck",
}


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VCAS-Stage203-Audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str) -> object:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_group(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.*?)(?:_?frame_|_?snapshot_)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def download_one(record: dict, images_root: Path) -> dict:
    filename = record["filename"]
    destination = images_root / filename
    expected_size = int(record["content_details"]["size"])
    expected_sha256 = record["content_details"]["sha256_hash"].lower()
    if destination.exists() and destination.is_file() and not destination.is_symlink():
        if destination.stat().st_size == expected_size and sha256_path(destination) == expected_sha256:
            return {"filename": filename, "downloaded": False}
    last_error = None
    for attempt in range(4):
        try:
            data = fetch_bytes(record["content_details"]["download_url"], timeout=180)
            break
        except Exception as error:  # network retry is bounded and existing verified files are reused
            last_error = error
            if attempt == 3:
                raise RuntimeError(f"download failed after 4 attempts for {filename}: {last_error}") from error
            time.sleep(2 ** attempt)
    if len(data) != expected_size:
        raise RuntimeError(f"size mismatch for {filename}: {len(data)} != {expected_size}")
    observed_sha256 = sha256_bytes(data)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"sha256 mismatch for {filename}: {observed_sha256} != {expected_sha256}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=str(images_root))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {"filename": filename, "downloaded": True}


def image_metrics(path: Path) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = ImageOps.grayscale(rgb)
        histogram = gray.histogram()
        pixels = rgb.width * rgb.height
        mean_luma = ImageStat.Stat(gray).mean[0]
        cumulative = 0
        p10 = 0
        target = math.ceil(pixels * 0.10)
        for value, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                p10 = value
                break
        dark_fraction = sum(histogram[:32]) / pixels
        return {
            "width": rgb.width,
            "height": rgb.height,
            "mean_luma": round(float(mean_luma), 4),
            "p10_luma": int(p10),
            "dark_fraction_below_32": round(float(dark_fraction), 6),
            "machine_lowlight_candidate": bool(mean_luma < 75 or dark_fraction >= 0.35),
            "machine_night_candidate": bool(mean_luma < 55 and dark_fraction >= 0.45),
        }


def make_contact_sheets(rows: list[dict], images_root: Path, output_root: Path) -> list[dict]:
    sheets = []
    columns = 5
    rows_per_sheet = 8
    cell_w, cell_h = 290, 205
    thumb_w, thumb_h = 280, 158
    font = ImageFont.load_default()
    ordered = sorted(rows, key=lambda item: (item["source_video_group"], item["filename"]))
    for page_index in range(math.ceil(len(ordered) / (columns * rows_per_sheet))):
        page_rows = ordered[page_index * columns * rows_per_sheet : (page_index + 1) * columns * rows_per_sheet]
        canvas = Image.new("RGB", (columns * cell_w, rows_per_sheet * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(page_rows):
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            with Image.open(images_root / row["filename"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (thumb_w, thumb_h))
            canvas.paste(thumb, (x + 5, y + 3))
            flag = "N?" if row["machine_night_candidate"] else ("L?" if row["machine_lowlight_candidate"] else "D")
            draw.text((x + 5, y + 164), f"{row['filename'][:42]}", fill="black", font=font)
            draw.text((x + 5, y + 179), f"group={row['source_video_group']} luma={row['mean_luma']:.1f} {flag}", fill="black", font=font)
        output_path = output_root / f"stage203-original-scenes-contact-{page_index + 1:02d}.jpg"
        canvas.save(output_path, quality=92)
        sheets.append({
            "path": str(output_path),
            "rows": len(page_rows),
            "sha256": sha256_path(output_path),
        })
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    forbidden_markers = ("vcas_rtsp_demo_60s", "36-48")
    for path in (source_root, output_root):
        if any(marker in str(path).lower() for marker in forbidden_markers):
            raise RuntimeError("frozen-video path marker detected")
    images_root = source_root / "original_images"
    images_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    folders = fetch_json(f"{PUBLIC_API}/folders/{DATASET_VERSION}")
    if not any(folder.get("id") == FOLDER_ID for folder in folders):
        raise RuntimeError("pinned source folder id not found")
    root_files = fetch_json(f"{PUBLIC_API}/files?folder_id=root&version={DATASET_VERSION}")
    folder_files = fetch_json(f"{PUBLIC_API}/files?folder_id={FOLDER_ID}&version={DATASET_VERSION}")
    label_record = next((item for item in root_files if item.get("filename") == EXPECTED_LABEL_NAME), None)
    if label_record is None:
        raise RuntimeError("sample_labels.csv not found")
    label_data = fetch_bytes(label_record["content_details"]["download_url"])
    if len(label_data) != EXPECTED_LABEL_BYTES or sha256_bytes(label_data) != EXPECTED_LABEL_SHA256:
        raise RuntimeError("sample_labels.csv integrity mismatch")
    labels_path = source_root / EXPECTED_LABEL_NAME
    labels_path.write_bytes(label_data)

    originals = sorted(
        (item for item in folder_files if not item["filename"].startswith("aug_")),
        key=lambda item: item["filename"],
    )
    if len(folder_files) != 1000 or len(originals) != EXPECTED_ORIGINAL_IMAGES:
        raise RuntimeError(f"unexpected source counts: all={len(folder_files)} originals={len(originals)}")
    if sum(int(item["content_details"]["size"]) for item in originals) != EXPECTED_ORIGINAL_BYTES:
        raise RuntimeError("unexpected original image byte total")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda item: download_one(item, images_root), originals))

    labels = list(csv.DictReader(io.StringIO(label_data.decode("utf-8-sig"))))
    labels_by_filename: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        labels_by_filename[row["filename"]].append(row)

    audit_rows = []
    exact_hashes = Counter()
    valid_box_rows = 0
    invalid_box_rows = 0
    exact_body_counts = Counter()
    all_class_counts = Counter()
    for record in originals:
        filename = record["filename"]
        path = images_root / filename
        observed_hash = sha256_path(path)
        if observed_hash != record["content_details"]["sha256_hash"].lower():
            raise RuntimeError(f"post-download sha256 mismatch for {filename}")
        exact_hashes[observed_hash] += 1
        metrics = image_metrics(path)
        source_rows = labels_by_filename.get(filename, [])
        for row in source_rows:
            klass = row["class"].strip().lower()
            all_class_counts[klass] += 1
            if klass in EXACT_BODY_MAP:
                exact_body_counts[EXACT_BODY_MAP[klass]] += 1
            try:
                xmin, ymin, xmax, ymax = (int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
                valid = 0 <= xmin < xmax <= metrics["width"] and 0 <= ymin < ymax <= metrics["height"]
            except (KeyError, ValueError):
                valid = False
            if valid:
                valid_box_rows += 1
            else:
                invalid_box_rows += 1
        audit_rows.append({
            "filename": filename,
            "source_video_group": video_group(filename),
            "file_id": record["id"],
            "bytes": int(record["content_details"]["size"]),
            "sha256": observed_hash,
            "label_rows": len(source_rows),
            **metrics,
        })

    manifest_path = output_root / "stage203-original-image-audit.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    sheets = make_contact_sheets(audit_rows, images_root, output_root)
    report = {
        "schema_version": "stage203-junction-original-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "component_download_and_machine_audit_complete_pending_agent_visual_scene_review",
        "source": {
            "title": "Junction-based Vehicle Detection Dataset",
            "doi": "10.17632/vwjg6b7kpt.1",
            "version": 1,
            "license": "CC BY 4.0",
            "landing_url": "https://data.mendeley.com/datasets/vwjg6b7kpt/1",
            "public_api": PUBLIC_API,
        },
        "scope": {
            "published_sample_images": len(folder_files),
            "source_original_images": len(originals),
            "source_augmented_images_excluded": len(folder_files) - len(originals),
            "original_bytes": sum(int(item["content_details"]["size"]) for item in originals),
            "source_video_groups": len({row["source_video_group"] for row in audit_rows}),
            "downloaded_this_run": sum(bool(item["downloaded"]) for item in results),
        },
        "integrity": {
            "all_images_readable": True,
            "all_per_file_sha256_verified": True,
            "exact_duplicate_images": sum(count - 1 for count in exact_hashes.values() if count > 1),
            "label_file_bytes": len(label_data),
            "label_file_sha256": sha256_bytes(label_data),
            "valid_box_rows": valid_box_rows,
            "invalid_box_rows": invalid_box_rows,
        },
        "labels": {
            "all_original_class_counts": dict(sorted(all_class_counts.items())),
            "provisional_exact_project_body_counts_before_crop_quality_teacher_and_cross_source_dedup": dict(sorted(exact_body_counts.items())),
            "coarse_or_out_of_scope_not_exact": ["car", "minibus", "motorcycle", "person", "taksi", "bicycle"],
            "color_truth_rows": 0,
        },
        "scene_machine_screen": {
            "lowlight_candidates": sum(row["machine_lowlight_candidate"] for row in audit_rows),
            "night_candidates": sum(row["machine_night_candidate"] for row in audit_rows),
            "semantics": "candidate_only; never confirmed natural night until full contact-sheet review and group evidence",
        },
        "outputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_path(manifest_path),
            "contact_sheets": sheets,
        },
        "policy": {
            "augmentations_count_toward_real_night_quota": False,
            "training_authorized": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = output_root / "stage203-junction-original-audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = [
        f"{sha256_path(report_path)}  {report_path.name}",
        f"{sha256_path(manifest_path)}  {manifest_path.name}",
    ]
    sums.extend(f"{item['sha256']}  {Path(item['path']).name}" for item in sheets)
    (output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "scope": report["scope"],
        "integrity": report["integrity"],
        "labels": report["labels"],
        "scene_machine_screen": report["scene_machine_screen"],
        "report": str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
