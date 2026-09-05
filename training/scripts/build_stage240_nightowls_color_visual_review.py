#!/usr/bin/env python3
"""Build fail-closed, track-level NightOwls color review material.

This stage does not create color labels.  It selects only train-split tracks with
enough independent, visible-spectrum evidence and renders three temporally
separated crops per track for later agent review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def choose_three(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = [row for row in rows if not row.get("internal_near_duplicate_of", "").strip()]
    unique.sort(key=lambda row: (int(row["frame_index"]), -float(row["quality_score"])))
    if len(unique) < 3:
        raise RuntimeError("track passed selection without three independent frames")
    positions = [0, (len(unique) - 1) // 2, len(unique) - 1]
    return [unique[index] for index in positions]


def fit_crop(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (18, 18, 18))
        canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-track-length", type=int, default=3)
    parser.add_argument("--minimum-independent-frames", type=int, default=3)
    parser.add_argument("--minimum-average-quality", type=float, default=0.04)
    parser.add_argument("--minimum-average-luma", type=float, default=40.0)
    parser.add_argument("--minimum-average-saturation", type=float, default=25.0)
    parser.add_argument("--minimum-width", type=int, default=48)
    parser.add_argument("--minimum-height", type=int, default=32)
    parser.add_argument("--tracks-per-page", type=int, default=20)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output_dir}")
    actual_manifest_sha = sha256(args.manifest)
    if actual_manifest_sha.lower() != args.expected_manifest_sha256.lower():
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_manifest_sha}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("Stage240 accepts only a non-empty train-split manifest")
    if any(row.get("source_dataset") != "NightOwls" for row in rows):
        raise RuntimeError("unexpected source dataset")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["track_key"]].append(row)

    selected: list[tuple[str, list[dict[str, str]], dict[str, float | int | str]]] = []
    rejected = Counter()
    for track_key, track_rows in sorted(grouped.items()):
        usable = [row for row in track_rows if truthy(row.get("crop_quality_usable", "false"))]
        if len(track_rows) < args.minimum_track_length or len(usable) < args.minimum_track_length:
            rejected["short_or_unusable_track"] += 1
            continue
        independent = [row for row in usable if not row.get("internal_near_duplicate_of", "").strip()]
        if len(independent) < args.minimum_independent_frames:
            rejected["insufficient_independent_frames"] += 1
            continue
        average_quality = sum(float(row["quality_score"]) for row in usable) / len(usable)
        average_luma = sum(float(row["crop_mean_luma"]) for row in usable) / len(usable)
        average_saturation = sum(float(row["crop_mean_saturation"]) for row in usable) / len(usable)
        minimum_width = min(int(row["crop_width"]) for row in usable)
        minimum_height = min(int(row["crop_height"]) for row in usable)
        if average_quality < args.minimum_average_quality:
            rejected["low_average_quality"] += 1
            continue
        if average_luma < args.minimum_average_luma:
            rejected["too_dark_for_reliable_color"] += 1
            continue
        if average_saturation < args.minimum_average_saturation:
            rejected["insufficient_chroma"] += 1
            continue
        if minimum_width < args.minimum_width or minimum_height < args.minimum_height:
            rejected["too_small_for_review"] += 1
            continue
        if all(truthy(row.get("border_truncated", "false")) for row in usable):
            rejected["all_frames_border_truncated"] += 1
            continue
        meta: dict[str, float | int | str] = {
            "track_length": len(track_rows),
            "usable_frames": len(usable),
            "independent_frames": len(independent),
            "average_quality": average_quality,
            "average_luma": average_luma,
            "average_saturation": average_saturation,
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
            "size_bin": Counter(row["size_bin"] for row in usable).most_common(1)[0][0],
        }
        selected.append((track_key, choose_three(usable), meta))

    args.output_dir.mkdir(parents=True)
    sheets_dir = args.output_dir / "contact-sheets"
    sheets_dir.mkdir()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 12) if font_path.is_file() else ImageFont.load_default()

    output_rows: list[dict[str, str]] = []
    tiles: list[tuple[str, Image.Image]] = []
    decoded = 0
    verified_hashes = 0
    for index, (track_key, representatives, meta) in enumerate(selected, start=1):
        audit_id = f"T{index:04d}"
        frame_images = []
        for representative in representatives:
            path = Path(representative["crop_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path).lower() != representative["crop_sha256"].lower():
                raise RuntimeError(f"crop SHA256 mismatch: {path}")
            verified_hashes += 1
            frame_images.append(fit_crop(path, (112, 84)))
            decoded += 1
            out = dict(representative)
            out.update(
                {
                    "stage240_audit_id": audit_id,
                    "stage240_review_role": "agent_visual_color_candidate_not_truth",
                    "stage240_average_quality": f"{float(meta['average_quality']):.8f}",
                    "stage240_average_luma": f"{float(meta['average_luma']):.6f}",
                    "stage240_average_saturation": f"{float(meta['average_saturation']):.6f}",
                    "stage240_independent_frames": str(meta["independent_frames"]),
                    "stage240_agent_color": "unknown",
                    "stage240_agent_decision": "pending",
                    "stage240_training_eligible": "false",
                }
            )
            output_rows.append(out)
        tile = Image.new("RGB", (352, 126), (245, 245, 245))
        for frame_index, frame in enumerate(frame_images):
            tile.paste(frame, (4 + frame_index * 116, 34))
        draw = ImageDraw.Draw(tile)
        parts = track_key.split("|")
        short_track = "/".join(parts[-3:])
        draw.text((4, 3), f"{audit_id} {short_track}", fill=(0, 0, 0), font=font)
        draw.text(
            (4, 19),
            f"{meta['size_bin']} L={float(meta['average_luma']):.0f} S={float(meta['average_saturation']):.0f} Q={float(meta['average_quality']):.3f}",
            fill=(20, 20, 20),
            font=small_font,
        )
        tiles.append((audit_id, tile))

    columns = 4
    rows_per_page = math.ceil(args.tracks_per_page / columns)
    sheet_paths: list[Path] = []
    for page_start in range(0, len(tiles), args.tracks_per_page):
        page_tiles = tiles[page_start : page_start + args.tracks_per_page]
        page = Image.new("RGB", (columns * 352, rows_per_page * 126), (225, 225, 225))
        for local_index, (_, tile) in enumerate(page_tiles):
            x = (local_index % columns) * 352
            y = (local_index // columns) * 126
            page.paste(tile, (x, y))
        page_number = page_start // args.tracks_per_page + 1
        sheet_path = sheets_dir / f"stage240-nightowls-color-tracks-page-{page_number:02d}.jpg"
        page.save(sheet_path, quality=94, subsampling=0)
        sheet_paths.append(sheet_path)

    manifest_path = args.output_dir / "stage240-nightowls-color-visual-candidates.csv"
    fieldnames = list(output_rows[0].keys()) if output_rows else list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema_version": "stage240-nightowls-color-visual-review-material-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_pending_agent_visual_color_audit",
        "input": {"manifest": str(args.manifest), "sha256": actual_manifest_sha, "rows": len(rows), "tracks": len(grouped)},
        "selection_thresholds": {
            "minimum_track_length": args.minimum_track_length,
            "minimum_independent_frames": args.minimum_independent_frames,
            "minimum_average_quality": args.minimum_average_quality,
            "minimum_average_luma": args.minimum_average_luma,
            "minimum_average_saturation": args.minimum_average_saturation,
            "minimum_width": args.minimum_width,
            "minimum_height": args.minimum_height,
            "not_all_frames_border_truncated": True,
        },
        "selected_tracks": len(selected),
        "selected_representative_rows": len(output_rows),
        "selected_size_counts": dict(sorted(Counter(str(meta["size_bin"]) for _, _, meta in selected).items())),
        "rejected_track_reason_counts": dict(sorted(rejected.items())),
        "decoded_representatives": decoded,
        "verified_crop_sha256": verified_hashes,
        "contact_sheets": len(sheet_paths),
        "outputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "contact_sheets": [{"path": str(path), "sha256": sha256(path)} for path in sheet_paths],
        },
        "policy": {
            "train_split_only": True,
            "visual_candidates_are_not_labels": True,
            "color_labels_generated": False,
            "training_started": False,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "research_only": True,
            "deployment_eligible": False,
        },
    }
    report_path = args.output_dir / "stage240-nightowls-color-visual-review-material.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = [args.manifest, manifest_path, report_path, *sheet_paths]
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir) if path.is_relative_to(args.output_dir) else path.name}\n" for path in sums),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
