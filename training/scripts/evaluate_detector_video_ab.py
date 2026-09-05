#!/usr/bin/env python3
"""Run two YOLO detectors on a frozen video and generate an auditable A/B report."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
import numpy as np


SEGMENTS = ((0, 12), (12, 24), (24, 36), (36, 48), (48, 60))
CONTACT_TIMES = (36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 47.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def segment_name(timestamp: float) -> str:
    for start, end in SEGMENTS:
        if start <= timestamp < end:
            return f"{start:02d}-{end:02d}s"
    return "outside"


def summarize_frames(frames: list[dict]) -> dict:
    counts = [len(row["detections"]) for row in frames]
    per_class: Counter[str] = Counter()
    segments: dict[str, list[int]] = {}
    segment_classes: dict[str, Counter[str]] = {}
    for row, count in zip(frames, counts, strict=True):
        name = segment_name(float(row["timestamp_s"]))
        segments.setdefault(name, []).append(count)
        segment_classes.setdefault(name, Counter())
        for detection in row["detections"]:
            per_class[detection["class_name"]] += 1
            segment_classes[name][detection["class_name"]] += 1

    def count_summary(values: list[int]) -> dict:
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0
        return {
            "frames": len(values),
            "zero_detection_frames": sum(value == 0 for value in values),
            "mean_detections_per_frame": round(mean(values), 4) if values else 0.0,
            "p95_detections_per_frame": p95,
            "detection_instances": sum(values),
        }

    return {
        **count_summary(counts),
        "class_detection_instances": dict(sorted(per_class.items())),
        "segments": {
            name: {
                **count_summary(values),
                "class_detection_instances": dict(sorted(segment_classes[name].items())),
            }
            for name, values in segments.items()
        },
    }


def run_model(
    model_path: Path,
    label: str,
    video_path: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[int, np.ndarray], dict]:
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_root / f"{label}-annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create annotated video")
    model = YOLO(str(model_path))
    frames: list[dict] = []
    contacts: dict[int, np.ndarray] = {}
    contact_indices = {round(value * fps): value for value in CONTACT_TIMES}
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    elapsed_inference = 0.0

    def flush() -> None:
        nonlocal elapsed_inference
        if not batch_frames:
            return
        started = time.perf_counter()
        results = model.predict(
            source=batch_frames,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=args.iou,
            device=args.device,
            half=True,
            verbose=False,
        )
        elapsed_inference += time.perf_counter() - started
        for frame_index, result in zip(batch_indices, results, strict=True):
            detections: list[dict] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for box, class_id, confidence in zip(boxes, classes, confidences, strict=True):
                    index = int(class_id)
                    detections.append(
                        {
                            "class_id": index,
                            "class_name": str(result.names[index]),
                            "confidence": round(float(confidence), 6),
                            "xyxy": [round(float(value), 3) for value in box],
                        }
                    )
            annotated = result.plot()
            writer.write(annotated)
            if frame_index in contact_indices:
                contacts[frame_index] = annotated.copy()
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 3),
                    "detections": detections,
                }
            )
        batch_frames.clear()
        batch_indices.clear()

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        batch_frames.append(frame)
        batch_indices.append(frame_index)
        frame_index += 1
        if len(batch_frames) >= args.batch:
            flush()
    flush()
    capture.release()
    writer.release()
    with (output_root / f"{label}-detections.jsonl").open("w", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    runtime = {
        "frames": len(frames),
        "inference_seconds": round(elapsed_inference, 4),
        "inference_ms_per_frame": round(elapsed_inference * 1000.0 / max(1, len(frames)), 4),
    }
    return frames, contacts, runtime


def compare_frames(old_frames: list[dict], new_frames: list[dict], threshold: float) -> dict:
    totals: Counter[str] = Counter()
    segment_totals: dict[str, Counter[str]] = {}
    for old, new in zip(old_frames, new_frames, strict=True):
        used: set[int] = set()
        matched = 0
        for old_detection in old["detections"]:
            best: tuple[float, int] | None = None
            for index, new_detection in enumerate(new["detections"]):
                if index in used or old_detection["class_name"] != new_detection["class_name"]:
                    continue
                overlap = box_iou(old_detection["xyxy"], new_detection["xyxy"])
                if overlap >= threshold and (best is None or overlap > best[0]):
                    best = (overlap, index)
            if best is not None:
                used.add(best[1])
                matched += 1
        row = Counter(
            matched=matched,
            old_only=len(old["detections"]) - matched,
            new_only=len(new["detections"]) - matched,
        )
        totals.update(row)
        name = segment_name(float(old["timestamp_s"]))
        segment_totals.setdefault(name, Counter()).update(row)
    return {
        "match_iou": threshold,
        **dict(totals),
        "segments": {name: dict(values) for name, values in segment_totals.items()},
    }


def write_contact_sheet(
    old_contacts: dict[int, np.ndarray],
    new_contacts: dict[int, np.ndarray],
    fps: float,
    output: Path,
) -> None:
    rows: list[np.ndarray] = []
    for index in sorted(set(old_contacts) & set(new_contacts)):
        left = cv2.resize(old_contacts[index], (640, 360))
        right = cv2.resize(new_contacts[index], (640, 360))
        cv2.putText(left, f"OLD {index / fps:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(right, f"NEW {index / fps:.1f}s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(cv2.hconcat([left, right]))
    if not rows or not cv2.imwrite(str(output), cv2.vconcat(rows)):
        raise RuntimeError("failed to write contact sheet")


def main() -> int:
    args = parse_args()
    video = args.video.resolve()
    old_model = args.old_model.resolve()
    new_model = args.new_model.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    old_frames, old_contacts, old_runtime = run_model(old_model, "old", video, output_root, args)
    new_frames, new_contacts, new_runtime = run_model(new_model, "new", video, output_root, args)
    if len(old_frames) != frame_count or len(new_frames) != frame_count:
        raise RuntimeError(
            f"frame count mismatch: video={frame_count} old={len(old_frames)} new={len(new_frames)}"
        )
    write_contact_sheet(old_contacts, new_contacts, fps, output_root / "36-48s-contact-sheet.jpg")
    report = {
        "schema_version": "1.0",
        "video": {
            "path": str(video),
            "sha256": sha256_file(video),
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
        },
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "nms_iou": args.iou,
            "batch": args.batch,
        },
        "old": {
            "model": str(old_model),
            "sha256": sha256_file(old_model),
            "runtime": old_runtime,
            "summary": summarize_frames(old_frames),
        },
        "new": {
            "model": str(new_model),
            "sha256": sha256_file(new_model),
            "runtime": new_runtime,
            "summary": summarize_frames(new_frames),
        },
        "comparison": compare_frames(old_frames, new_frames, args.match_iou),
        "interpretation_limit": "A/B differences are not ground truth; visual review and fixed labels are required before release.",
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
