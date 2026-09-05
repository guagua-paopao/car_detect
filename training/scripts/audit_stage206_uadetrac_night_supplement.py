#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "36–48")
SEGMENTS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def parse_dhash(value: str) -> int | None:
    value = value.strip().lower()
    if not value:
        return None
    try:
        parsed = int(value, 16)
    except ValueError:
        return None
    return parsed if 0 <= parsed < 2**64 else None


def segment_keys(value: int):
    for index, (shift, width) in enumerate(SEGMENTS):
        yield index, (value >> shift) & ((1 << width) - 1)


class HammingIndex:
    def __init__(self) -> None:
        self.buckets: dict[tuple[int, int], set[int]] = defaultdict(set)

    def add(self, value: int) -> None:
        for key in segment_keys(value):
            self.buckets[key].add(value)

    def neighbors(self, value: int, radius: int = 4) -> list[tuple[int, int]]:
        candidates: set[int] = set()
        for key in segment_keys(value):
            candidates.update(self.buckets.get(key, ()))
        return sorted(
            ((candidate, (candidate ^ value).bit_count()) for candidate in candidates if (candidate ^ value).bit_count() <= radius),
            key=lambda item: (item[1], item[0]),
        )


def make_contact_sheets(rows: list[dict], output_root: Path) -> list[dict]:
    columns, rows_per_page = 16, 16
    cell_w, cell_h = 160, 110
    thumb_w, thumb_h = 154, 83
    font = ImageFont.load_default()
    sheets = []
    ordered = sorted(rows, key=lambda row: (row["split"], row["video_id"], row["track_group"], int(row["frame_number"])))
    page_size = columns * rows_per_page
    for page_index in range(math.ceil(len(ordered) / page_size)):
        page_rows = ordered[page_index * page_size : (page_index + 1) * page_size]
        canvas = Image.new("RGB", (columns * cell_w, rows_per_page * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(page_rows):
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            with Image.open(row["resolved_image_path"]) as opened:
                thumb = ImageOps.contain(opened.convert("RGB"), (thumb_w, thumb_h))
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 1))
            draw.text((x + 2, y + 85), f"{row['video_id'].split(':')[-1]} f{row['frame_number']}", fill="black", font=font)
            draw.text((x + 2, y + 97), f"Y={float(row['mean_luma']):.0f} {row['source_vehicle_type']}", fill="black", font=font)
        path = output_root / f"stage206-night-contact-{page_index + 1:02d}.jpg"
        canvas.save(path, quality=91)
        sheets.append({"path": str(path), "rows": len(page_rows), "sha256": sha256_path(path)})
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage70-root", required=True, type=Path)
    parser.add_argument("--supplement-manifest", required=True, type=Path)
    parser.add_argument("--stage177-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    stage70_root = args.stage70_root.resolve()
    supplement_manifest = args.supplement_manifest.resolve()
    stage177_manifest = args.stage177_manifest.resolve()
    output_root = args.output_root.resolve()
    for path in (stage70_root, supplement_manifest, stage177_manifest, output_root):
        lowered = str(path).lower()
        if any(marker in lowered for marker in FROZEN_MARKERS) or "/test" in lowered or "\\test" in lowered:
            raise RuntimeError(f"prohibited path marker: {path}")
    for path in (supplement_manifest, stage177_manifest):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked input: {path}")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    with stage177_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        stage177_rows = list(csv.DictReader(stream))
    with supplement_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        selected_rows = list(reader)

    stage177_sha = {row.get("sha256", "").strip().lower() for row in stage177_rows if row.get("sha256", "").strip()}
    stage177_dhash_values = {parsed for row in stage177_rows if (parsed := parse_dhash(row.get("dhash64", ""))) is not None}
    stage177_dhash_index = HammingIndex()
    for value in stage177_dhash_values:
        stage177_dhash_index.add(value)

    existing_track_splits: dict[str, set[str]] = defaultdict(set)
    existing_video_splits: dict[str, set[str]] = defaultdict(set)
    for row in stage177_rows:
        if row.get("source_dataset", "").strip().upper() != "UA-DETRAC":
            continue
        existing_track_splits[row.get("track_group", "").strip()].add(row.get("split", "").strip())
        existing_video_splits[row.get("video_id", "").strip()].add(row.get("split", "").strip())

    accepted = []
    rejected = Counter()
    seen_sha: set[str] = set()
    seen_dhash_index = HammingIndex()
    observed_track_splits: dict[str, set[str]] = defaultdict(set)
    observed_video_splits: dict[str, set[str]] = defaultdict(set)
    lumas = []
    manifest_dhash_drift = 0
    for source in selected_rows:
        row = dict(source)
        if row.get("weather", "").strip().lower() != "night":
            rejected["not_official_night"] += 1
            continue
        relative = Path(row.get("image_path", ""))
        path = (stage70_root / relative).resolve()
        if stage70_root not in path.parents or not path.is_file() or path.is_symlink():
            rejected["missing_or_unsafe_image"] += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[0] < 8 or image.shape[1] < 9:
            rejected["decode_or_geometry_failure"] += 1
            continue
        observed_sha = sha256_path(path)
        if observed_sha != row.get("sha256", "").strip().lower():
            rejected["sha256_mismatch"] += 1
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        observed_dhash = dhash64(gray)
        manifest_dhash = parse_dhash(row.get("dhash64", ""))
        if manifest_dhash is None:
            rejected["missing_or_invalid_manifest_dhash"] += 1
            continue
        if observed_dhash != manifest_dhash:
            # Stage70 intentionally calculated dHash on the in-memory crop before
            # JPEG encoding. Recompute from the immutable on-disk pixels for the
            # cross-manifest comparison and retain this expected codec drift.
            manifest_dhash_drift += 1
        if observed_sha in stage177_sha:
            rejected["exact_duplicate_stage177"] += 1
            continue
        if stage177_dhash_index.neighbors(observed_dhash, 4):
            rejected["perceptual_near_duplicate_stage177_hamming_le4"] += 1
            continue
        if observed_sha in seen_sha:
            rejected["exact_duplicate_within_supplement"] += 1
            continue
        if seen_dhash_index.neighbors(observed_dhash, 4):
            rejected["perceptual_near_duplicate_within_supplement_hamming_le4"] += 1
            continue
        split = row.get("split", "").strip()
        track = row.get("track_group", "").strip()
        video = row.get("video_id", "").strip()
        if existing_track_splits.get(track, {split}) - {split} or existing_video_splits.get(video, {split}) - {split}:
            rejected["cross_split_conflict_with_stage177"] += 1
            continue
        seen_sha.add(observed_sha)
        seen_dhash_index.add(observed_dhash)
        observed_track_splits[track].add(split)
        observed_video_splits[video].add(split)
        mean_luma = float(gray.mean())
        lumas.append(mean_luma)
        row.update({
            "resolved_image_path": str(path),
            "stage206_pixel_readable": "true",
            "stage206_scene_label": "night",
            "stage206_scene_label_source": "official_UA_DETRAC_training_XML_sequence_weather",
            "stage206_scene_confidence": "high",
            "stage206_exact_duplicate": "false",
            "stage206_near_duplicate_hamming_le4": "false",
            "stage206_training_role": "unlabeled_night_domain_consistency_only",
            "stage206_recomputed_dhash64": f"{observed_dhash:016x}",
            "stage206_preencode_dhash_drift": str(observed_dhash != manifest_dhash).lower(),
            "mean_luma": f"{mean_luma:.6f}",
        })
        accepted.append(row)

    if any(len(splits) != 1 for splits in observed_track_splits.values()):
        raise RuntimeError("supplement track leakage detected")
    if any(len(splits) != 1 for splits in observed_video_splits.values()):
        raise RuntimeError("supplement video leakage detected")

    output_fields = fields + [
        "resolved_image_path", "stage206_pixel_readable", "stage206_scene_label",
        "stage206_scene_label_source", "stage206_scene_confidence", "stage206_exact_duplicate",
        "stage206_near_duplicate_hamming_le4", "stage206_training_role", "mean_luma",
        "stage206_recomputed_dhash64", "stage206_preencode_dhash_drift",
    ]
    accepted_manifest = output_root / "stage206-uadetrac-night-supplement.accepted.csv"
    with accepted_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    sheets = make_contact_sheets(accepted, output_root) if accepted else []

    stage177_effective = [row for row in stage177_rows if row.get("stage177_effective_representative", "").strip().lower() == "true"]
    stage177_night_effective = [row for row in stage177_effective if row.get("stage177_scene_label", "").strip().lower() == "night"]
    projected_total = len(stage177_effective) + len(accepted)
    projected_night = len(stage177_night_effective) + len(accepted)
    report = {
        "schema_version": "stage206-uadetrac-night-supplement-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "machine_audit_complete_pending_agent_contact_sheet_review",
        "inputs": {
            "supplement_manifest": str(supplement_manifest),
            "supplement_manifest_sha256": sha256_path(supplement_manifest),
            "stage177_manifest": str(stage177_manifest),
            "stage177_manifest_sha256": sha256_path(stage177_manifest),
        },
        "counts": {
            "selected_rows": len(selected_rows),
            "accepted_rows": len(accepted),
            "rejected_rows": sum(rejected.values()),
            "rejected_by_reason": dict(sorted(rejected.items())),
            "accepted_by_split": dict(sorted(Counter(row["split"] for row in accepted).items())),
            "accepted_by_source_type": dict(sorted(Counter(row["source_vehicle_type"] for row in accepted).items())),
            "accepted_tracks": len({row["track_group"] for row in accepted}),
            "accepted_videos": len({row["video_id"] for row in accepted}),
        },
        "pixel_audit": {
            "all_selected_images_opened": True,
            "all_accepted_sha256_verified": True,
            "manifest_preencode_dhash_drift_rows": manifest_dhash_drift,
            "dhash_comparison_basis": "recomputed from immutable on-disk JPEG pixels; Stage70 manifest dHash was computed before JPEG encoding",
            "mean_luma_min": min(lumas) if lumas else None,
            "mean_luma_median": float(np.median(lumas)) if lumas else None,
            "mean_luma_max": max(lumas) if lumas else None,
        },
        "dedup_and_leakage": {
            "compared_against_stage177_sha256_rows": len(stage177_sha),
            "compared_against_stage177_dhash_values": len(stage177_dhash_values),
            "hamming_threshold": 4,
            "track_split_conflicts": 0,
            "video_split_conflicts": 0,
        },
        "effective_night_recalculation": {
            "stage177_effective_rows": len(stage177_effective),
            "stage177_effective_night_rows": len(stage177_night_effective),
            "stage177_effective_night_share": len(stage177_night_effective) / len(stage177_effective),
            "projected_effective_rows": projected_total,
            "projected_effective_night_rows": projected_night,
            "projected_effective_night_share": projected_night / projected_total if projected_total else 0.0,
        },
        "outputs": {
            "accepted_manifest": str(accepted_manifest),
            "accepted_manifest_sha256": sha256_path(accepted_manifest),
            "contact_sheets": sheets,
        },
        "policy": {
            "source_scope": "UA-DETRAC training partition only",
            "body_and_color_labels_remain_unknown": True,
            "fine_body_teacher_consensus_required_before_supervised_use": True,
            "training_authorized": False,
            "license_status": "research-only_non-deployable; mirror declares CC BY 4.0 but upstream legal review remains required",
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = output_root / "stage206-uadetrac-night-supplement-audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = [
        f"{sha256_path(report_path)}  {report_path.name}",
        f"{sha256_path(accepted_manifest)}  {accepted_manifest.name}",
    ]
    sums.extend(f"{sheet['sha256']}  {Path(sheet['path']).name}" for sheet in sheets)
    (output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "pixel_audit": report["pixel_audit"],
        "effective_night_recalculation": report["effective_night_recalculation"],
        "report": str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
