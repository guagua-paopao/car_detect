#!/usr/bin/env python3
"""Mine conservative low-light track rows from existing train crops only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def analyze(item: tuple[int, str]) -> tuple[int, dict]:
    index, path_string = item
    image = cv2.imread(path_string, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return index, {"error": "unreadable"}
    height, width = image.shape[:2]
    scale = min(1.0, 192.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    border_y = max(2, round(height * 0.12))
    border_x = max(2, round(width * 0.12))
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True
    mean = float(gray.mean())
    median = float(np.median(gray))
    p90 = float(np.percentile(gray, 90))
    border_mean = float(gray[border_mask].mean())
    dark_fraction = float((gray < 55.0).mean())
    highlight_fraction = float((gray > 235.0).mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    darkness_mean = float(np.clip((95.0 - mean) / 65.0, 0.0, 1.0))
    darkness_median = float(np.clip((90.0 - median) / 65.0, 0.0, 1.0))
    darkness_border = float(np.clip((100.0 - border_mean) / 70.0, 0.0, 1.0))
    darkness_p90 = float(np.clip((165.0 - p90) / 105.0, 0.0, 1.0))
    score = (
        0.28 * darkness_mean
        + 0.22 * darkness_median
        + 0.25 * darkness_border
        + 0.15 * min(1.0, dark_fraction / 0.60)
        + 0.10 * darkness_p90
    )
    provisional = (
        score >= 0.62
        and mean <= 88.0
        and median <= 82.0
        and border_mean <= 95.0
        and dark_fraction >= 0.30
    )
    return index, {
        "mean_luma": mean,
        "median_luma": median,
        "p90_luma": p90,
        "border_mean_luma": border_mean,
        "dark_fraction": dark_fraction,
        "highlight_fraction": highlight_fraction,
        "contrast": contrast,
        "blur_laplacian": blur,
        "lowlight_score": score,
        "provisional_lowlight": provisional,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--minimum-track-frames", type=int, default=3)
    parser.add_argument("--minimum-track-positive-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-track-median-score", type=float, default=0.65)
    parser.add_argument("--max-rows-per-track", type=int, default=5)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("refusing to overwrite low-light mining evidence")
    allowed_root = args.allowed_root.resolve()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        train_rows = []
        for row_number, row in enumerate(reader, 2):
            if row.get("split") != "train":
                continue
            if row.get("review_status") != "approved":
                continue
            if not truthy(row.get("formal_train_eligible")):
                continue
            if not truthy(row.get("body_type_supervised")) and not truthy(row.get("color_supervised")):
                continue
            absolute = (args.manifest.parent / row["image_path"]).resolve()
            try:
                absolute.relative_to(allowed_root)
            except ValueError as error:
                raise RuntimeError(f"train crop outside allowed root at row {row_number}") from error
            train_rows.append({**row, "_absolute_path": str(absolute), "_row_number": row_number})
    if not train_rows:
        raise RuntimeError("no eligible train rows for low-light mining")

    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "scan-state.json"
    state = {
        "schema_version": "existing-train-lowlight-scan-state-v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "eligible_train_rows": len(train_rows),
        "rows_scanned": 0,
        "workers": max(1, min(args.workers, 12)),
        "validation_or_test_images_opened": 0,
        "frozen_video_used": False,
    }
    atomic_json(state_path, state)
    results: list[dict | None] = [None] * len(train_rows)
    workers = state["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(analyze, (index, row["_absolute_path"])): index
            for index, row in enumerate(train_rows)
        }
        completed = 0
        for future in as_completed(futures):
            index, metrics = future.result()
            results[index] = metrics
            completed += 1
            if completed % 5000 == 0 or completed == len(train_rows):
                state.update(
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    rows_scanned=completed,
                    progress=completed / len(train_rows),
                )
                atomic_json(state_path, state)
                print(json.dumps({
                    "rows_scanned": completed,
                    "rows_total": len(train_rows),
                    "progress": state["progress"],
                }), flush=True)

    scan_rows = []
    unreadable = 0
    by_track: dict[str, list[dict]] = defaultdict(list)
    known_counts = Counter()
    source_provisional = Counter()
    for row, metrics in zip(train_rows, results):
        assert metrics is not None
        if "error" in metrics:
            unreadable += 1
            continue
        known_positive = (
            row.get("lighting") in {"low_light", "night"}
            or truthy(row.get("low_light"))
            or truthy(row.get("night"))
        )
        known_negative = (
            row.get("lighting") in {"daylight", "moderate_light"}
            and not truthy(row.get("night"))
        )
        provisional = bool(metrics["provisional_lowlight"])
        if known_positive:
            known_counts["tp" if provisional else "fn"] += 1
        elif known_negative:
            known_counts["fp" if provisional else "tn"] += 1
        if provisional:
            source_provisional[row.get("source_dataset", "unknown")] += 1
        output = {
            "image_path": row["image_path"],
            "source_dataset": row.get("source_dataset", ""),
            "camera_id": row.get("camera_id", ""),
            "video_id": row.get("video_id", ""),
            "track_group": row.get("track_group", ""),
            "source_frame_id": row.get("source_frame_id", ""),
            "body_type": row.get("body_type", ""),
            "color": row.get("color", ""),
            "current_lighting": row.get("lighting", ""),
            "current_low_light": row.get("low_light", ""),
            "current_night": row.get("night", ""),
            **{
                key: (str(value).lower() if isinstance(value, bool) else f"{value:.6f}")
                for key, value in metrics.items()
            },
        }
        scan_rows.append(output)
        track = row.get("track_group") or row.get("video_id") or row["image_path"]
        by_track[track].append({"source": row, "metrics": metrics, "scan": output})

    scan_path = args.output_root / "train-photometric-scan.csv"
    scan_fields = list(scan_rows[0]) if scan_rows else []
    with scan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scan_fields)
        writer.writeheader()
        writer.writerows(scan_rows)

    accepted_tracks = set()
    track_summaries = []
    for track, items in by_track.items():
        scores = sorted(float(item["metrics"]["lowlight_score"]) for item in items)
        positives = sum(bool(item["metrics"]["provisional_lowlight"]) for item in items)
        fraction = positives / len(items)
        median_score = scores[len(scores) // 2]
        accepted = (
            len(items) >= args.minimum_track_frames
            and fraction >= args.minimum_track_positive_fraction
            and median_score >= args.minimum_track_median_score
        )
        if accepted:
            accepted_tracks.add(track)
        track_summaries.append({
            "track_group": track,
            "frames": len(items),
            "provisional_positive_frames": positives,
            "positive_fraction": f"{fraction:.6f}",
            "median_lowlight_score": f"{median_score:.6f}",
            "accepted_lowlight_track": str(accepted).lower(),
        })
    track_summaries.sort(key=lambda row: row["track_group"])
    track_path = args.output_root / "track-lowlight-summary.csv"
    with track_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(track_summaries[0]))
        writer.writeheader()
        writer.writerows(track_summaries)

    overlay_rows = []
    overlay_source_counts = Counter()
    for track in sorted(accepted_tracks):
        candidates = [
            item for item in by_track[track]
            if bool(item["metrics"]["provisional_lowlight"])
        ]
        candidates.sort(key=lambda item: (
            -float(item["metrics"]["lowlight_score"]),
            item["source"]["image_path"],
        ))
        for item in candidates[: args.max_rows_per_track]:
            row = item["source"]
            metrics = item["metrics"]
            overlay_rows.append({
                "image_path": row["image_path"],
                "track_group": row.get("track_group", ""),
                "source_dataset": row.get("source_dataset", ""),
                "lighting": "low_light_proxy_multiframe",
                "low_light": "true",
                "lowlight_score": f"{float(metrics['lowlight_score']):.6f}",
                "photometric_mean": f"{float(metrics['mean_luma']):.6f}",
                "crop_mean_luma": f"{float(metrics['mean_luma']):.6f}",
                "crop_sharpness": f"{float(metrics['blur_laplacian']):.6f}",
                "hard_example_priority": "high",
                "hard_mining_tag": "existing_real_multiframe_lowlight_proxy",
                "hard_score_increment": "2.000000",
                "sample_weight_floor": "1.250000",
            })
            overlay_source_counts[row.get("source_dataset", "unknown")] += 1
    overlay_path = args.output_root / "lowlight-train-overlay.csv"
    overlay_fields = list(overlay_rows[0]) if overlay_rows else [
        "image_path", "track_group", "source_dataset", "lighting", "low_light",
        "lowlight_score", "photometric_mean", "crop_mean_luma", "crop_sharpness",
        "hard_example_priority", "hard_mining_tag", "hard_score_increment",
        "sample_weight_floor",
    ]
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overlay_fields)
        writer.writeheader()
        writer.writerows(overlay_rows)

    tp, fp = known_counts["tp"], known_counts["fp"]
    fn, tn = known_counts["fn"], known_counts["tn"]
    known_precision = tp / (tp + fp) if tp + fp else 0.0
    known_recall = tp / (tp + fn) if tp + fn else 0.0
    gates = {
        "minimum_1000_overlay_rows": len(overlay_rows) >= 1000,
        "minimum_200_accepted_tracks": len(accepted_tracks) >= 200,
        "known_train_precision_at_least_0_85": known_precision >= 0.85,
        "known_train_recall_at_least_0_25": known_recall >= 0.25,
        "all_images_train_only": True,
        "no_validation_or_test_images_opened": True,
    }
    status = "pass" if all(gates.values()) else "fail_threshold_or_quantity_gate"
    report = {
        "schema_version": "existing-train-lowlight-track-mining-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "eligible_train_rows": len(train_rows),
        "scanned_rows": len(scan_rows),
        "unreadable_rows": unreadable,
        "provisional_lowlight_rows_by_source": dict(sorted(source_provisional.items())),
        "tracks_scanned": len(by_track),
        "accepted_lowlight_tracks": len(accepted_tracks),
        "overlay_rows": len(overlay_rows),
        "overlay_rows_by_source": dict(sorted(overlay_source_counts.items())),
        "known_train_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "known_train_precision": known_precision,
        "known_train_recall": known_recall,
        "thresholds": {
            "frame_score": 0.62,
            "frame_mean_luma_max": 88.0,
            "frame_median_luma_max": 82.0,
            "frame_border_mean_luma_max": 95.0,
            "frame_dark_fraction_min": 0.30,
            "minimum_track_frames": args.minimum_track_frames,
            "minimum_track_positive_fraction": args.minimum_track_positive_fraction,
            "minimum_track_median_score": args.minimum_track_median_score,
            "max_rows_per_track": args.max_rows_per_track,
        },
        "gates": gates,
        "scan_csv": str(scan_path.resolve()),
        "scan_csv_sha256": sha256(scan_path),
        "track_summary_csv": str(track_path.resolve()),
        "track_summary_csv_sha256": sha256(track_path),
        "overlay_csv": str(overlay_path.resolve()),
        "overlay_csv_sha256": sha256(overlay_path),
        "policy": {
            "train_rows_only": True,
            "labels_unchanged": True,
            "metadata_and_sampling_overlay_only": True,
            "minimum_three_frame_track_consistency": True,
            "validation_or_test_images_opened": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "apply overlay only on pass; otherwise tune thresholds using train metadata evidence and rerun into a new output directory",
    }
    report_path = args.output_root / "lowlight-mining-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.update(
        status="complete" if status == "pass" else "complete_fail_closed",
        updated_at=datetime.now(timezone.utc).isoformat(),
        rows_scanned=len(scan_rows),
        progress=1.0,
        report=str(report_path.resolve()),
        report_sha256=sha256(report_path),
    )
    atomic_json(state_path, state)
    print(json.dumps({
        "status": status,
        "scanned": len(scan_rows),
        "unreadable": unreadable,
        "tracks": len(by_track),
        "accepted_tracks": len(accepted_tracks),
        "overlay_rows": len(overlay_rows),
        "known_precision": known_precision,
        "known_recall": known_recall,
        "gates": gates,
    }, ensure_ascii=False), flush=True)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
