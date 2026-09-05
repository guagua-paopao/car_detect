#!/usr/bin/env python3
"""Reconstruct conservative BMD-45 train-only vehicle tracks.

The legacy attribute manifest grouped all boxes from one frame together.  This
tool instead links individual BMD-45 COCO boxes over time using source class,
motion, box geometry, and crop appearance.  It only emits evidence; it never
creates color labels or reads validation/test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


TYPE_MAP = {
    1: "sedan",
    2: "suv",
    3: "mpv",
    4: "bus",
    5: "heavy_truck",
    8: "light_truck",
    9: "bus",
    10: "bus",
    12: "van",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_number(file_name: str) -> int:
    stem = Path(file_name).stem
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(f"BMD-45 frame filename is not numeric: {file_name}") from exc


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    la = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    ra = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = la + ra - inter
    return inter / union if union > 0 else 0.0


def center_and_size(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (box[:2] + box[2:]) * 0.5, np.maximum(box[2:] - box[:2], 1.0)


def appearance_histogram(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read crop: {path}")
    height, width = image.shape[:2]
    x0, x1 = int(width * 0.08), max(int(width * 0.92), 1)
    y0, y1 = int(height * 0.08), max(int(height * 0.92), 1)
    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).reshape(-1)
    value = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
    hist = np.concatenate([hs, value]).astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


@dataclass
class Detection:
    sequence: str
    frame: int
    image_id: int
    annotation_id: int
    source_category_id: int
    source_category: str
    body_type: str
    image_path: str
    bbox: np.ndarray
    image_width: int
    image_height: int
    hist: np.ndarray
    match_cost: float = 0.0
    hist_distance: float = 0.0
    center_distance: float = 0.0


@dataclass
class Track:
    track_id: str
    source_category_id: int
    items: list[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.items[-1]

    def predicted_bbox(self, target_frame: int) -> np.ndarray:
        last = self.items[-1]
        if len(self.items) < 2:
            return last.bbox.copy()
        previous = self.items[-2]
        delta = max(1, last.frame - previous.frame)
        horizon = max(0, target_frame - last.frame)
        last_center, last_size = center_and_size(last.bbox)
        prev_center, prev_size = center_and_size(previous.bbox)
        center = last_center + (last_center - prev_center) * (horizon / delta)
        size = np.maximum(1.0, last_size + (last_size - prev_size) * (horizon / delta))
        return np.concatenate([center - size * 0.5, center + size * 0.5])


def association(track: Track, detection: Detection, max_frame_gap: int) -> tuple[float, float, float] | None:
    gap = detection.frame - track.last.frame
    if gap <= 0 or gap > max_frame_gap or detection.source_category_id != track.source_category_id:
        return None
    predicted = track.predicted_bbox(detection.frame)
    pc, ps = center_and_size(predicted)
    dc, ds = center_and_size(detection.bbox)
    scale = max(float(np.linalg.norm(ps)), float(np.linalg.norm(ds)), 1.0)
    center_distance = float(np.linalg.norm(pc - dc) / scale)
    size_distance = float(np.max(np.abs(np.log(np.maximum(ds, 1.0) / np.maximum(ps, 1.0)))))
    hist_distance = float(cv2.compareHist(track.last.hist, detection.hist, cv2.HISTCMP_BHATTACHARYYA))
    overlap = bbox_iou(predicted, detection.bbox)
    # With a median source-frame gap near 20, overlap can be zero.  The first
    # link is therefore stricter on appearance; later motion-predicted links
    # can tolerate a little more displacement.
    center_gate = 1.65 if len(track.items) < 2 else 1.25
    if center_distance > center_gate or size_distance > 0.90 or hist_distance > 0.48:
        return None
    cost = 0.43 * min(center_distance / center_gate, 1.0)
    cost += 0.17 * (1.0 - overlap)
    cost += 0.32 * min(hist_distance / 0.48, 1.0)
    cost += 0.08 * min(size_distance / 0.90, 1.0)
    # Longer gaps are possible but should lose against a similar nearby box.
    cost += 0.08 * min(gap / max_frame_gap, 1.0)
    return cost, hist_distance, center_distance


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-frame-gap", type=int, default=90)
    parser.add_argument("--max-match-cost", type=float, default=0.82)
    parser.add_argument("--minimum-track-frames", type=int, default=3)
    parser.add_argument("--maximum-track-median-hist-distance", type=float, default=0.30)
    parser.add_argument("--maximum-track-p90-match-cost", type=float, default=0.78)
    args = parser.parse_args()
    for path in (args.output_csv, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    data = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in data["categories"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    skipped = Counter()
    for annotation in data["annotations"]:
        category_id = int(annotation.get("category_id", -1))
        if category_id not in TYPE_MAP:
            skipped[f"unmapped_category:{categories.get(category_id, 'unknown')}"] += 1
            continue
        by_image[int(annotation["image_id"])].append(annotation)

    sequence_images: dict[str, list[dict]] = defaultdict(list)
    for image in images.values():
        sequence_images[Path(str(image["file_name"])).parent.name].append(image)

    finished_tracks: list[Track] = []
    missing_crops: list[str] = []
    detections_seen = 0
    for sequence, sequence_rows in sorted(sequence_images.items()):
        sequence_rows.sort(key=lambda item: frame_number(str(item["file_name"])))
        active: list[Track] = []
        next_track = 0
        for image in sequence_rows:
            current_frame = frame_number(str(image["file_name"]))
            retained: list[Track] = []
            for track in active:
                if current_frame - track.last.frame <= args.max_frame_gap:
                    retained.append(track)
                else:
                    finished_tracks.append(track)
            active = retained

            detections: list[Detection] = []
            image_id = int(image["id"])
            for annotation in by_image.get(image_id, []):
                crop_name = f"bmdraw_{image_id}_{int(annotation['id'])}.jpg"
                crop_path = args.crop_root / "crops" / "train" / crop_name
                if not crop_path.is_file():
                    skipped["mapped_crop_missing"] += 1
                    if len(missing_crops) < 20:
                        missing_crops.append(str(crop_path))
                    continue
                x, y, width, height = (float(value) for value in annotation["bbox"])
                if width < 12 or height < 12:
                    skipped["tiny_box"] += 1
                    continue
                category_id = int(annotation["category_id"])
                detections.append(Detection(
                    sequence=sequence,
                    frame=current_frame,
                    image_id=image_id,
                    annotation_id=int(annotation["id"]),
                    source_category_id=category_id,
                    source_category=categories[category_id],
                    body_type=TYPE_MAP[category_id],
                    image_path=str(crop_path.resolve()),
                    bbox=np.asarray([x, y, x + width, y + height], dtype=np.float32),
                    image_width=int(image["width"]),
                    image_height=int(image["height"]),
                    hist=appearance_histogram(crop_path),
                ))
            detections_seen += len(detections)

            if active and detections:
                costs = np.full((len(active), len(detections)), 1e6, dtype=np.float64)
                evidence: dict[tuple[int, int], tuple[float, float, float]] = {}
                for track_index, track in enumerate(active):
                    for detection_index, detection in enumerate(detections):
                        result = association(track, detection, args.max_frame_gap)
                        if result is not None:
                            costs[track_index, detection_index] = result[0]
                            evidence[(track_index, detection_index)] = result
                matched_tracks: set[int] = set()
                matched_detections: set[int] = set()
                for track_index, detection_index in zip(*linear_sum_assignment(costs)):
                    cost = float(costs[track_index, detection_index])
                    if cost > args.max_match_cost or (track_index, detection_index) not in evidence:
                        continue
                    _, hist_distance, center_distance = evidence[(track_index, detection_index)]
                    detection = detections[detection_index]
                    detection.match_cost = cost
                    detection.hist_distance = hist_distance
                    detection.center_distance = center_distance
                    active[track_index].items.append(detection)
                    matched_tracks.add(track_index)
                    matched_detections.add(detection_index)
                unmatched = [d for index, d in enumerate(detections) if index not in matched_detections]
            else:
                unmatched = detections

            for detection in unmatched:
                track_id = f"BMD45T:{sequence}:{next_track:06d}"
                next_track += 1
                active.append(Track(track_id=track_id, source_category_id=detection.source_category_id, items=[detection]))
        finished_tracks.extend(active)

    output_rows: list[dict[str, str]] = []
    valid_tracks = 0
    valid_rows = 0
    valid_type_counts = Counter()
    length_histogram = Counter()
    track_median_hist_distances: list[float] = []
    track_p90_match_costs: list[float] = []
    for track in finished_tracks:
        length = len(track.items)
        length_histogram[length] += 1
        hist_distances = [item.hist_distance for item in track.items[1:]]
        match_costs = [item.match_cost for item in track.items[1:]]
        median_hist = statistics.median(hist_distances) if hist_distances else 0.0
        p90_cost = percentile(match_costs, 0.90) or 0.0
        frame_gaps = [right.frame - left.frame for left, right in zip(track.items, track.items[1:])]
        track_valid = (
            length >= args.minimum_track_frames
            and median_hist <= args.maximum_track_median_hist_distance
            and p90_cost <= args.maximum_track_p90_match_cost
        )
        if track_valid:
            valid_tracks += 1
            valid_rows += length
            valid_type_counts[track.items[0].body_type] += 1
            track_median_hist_distances.append(float(median_hist))
            track_p90_match_costs.append(float(p90_cost))
        for position, item in enumerate(track.items):
            box = item.bbox
            area_ratio = float((box[2] - box[0]) * (box[3] - box[1]) / (item.image_width * item.image_height))
            output_rows.append({
                "sequence": item.sequence,
                "frame_number": str(item.frame),
                "image_id": str(item.image_id),
                "annotation_id": str(item.annotation_id),
                "track_id": track.track_id,
                "track_position": str(position),
                "track_length": str(length),
                "track_valid": str(track_valid).lower(),
                "source_category_id": str(item.source_category_id),
                "source_category": item.source_category,
                "body_type": item.body_type,
                "image_path": item.image_path,
                "bbox_x1": f"{box[0]:.3f}",
                "bbox_y1": f"{box[1]:.3f}",
                "bbox_x2": f"{box[2]:.3f}",
                "bbox_y2": f"{box[3]:.3f}",
                "bbox_area_ratio": f"{area_ratio:.8f}",
                "match_cost": f"{item.match_cost:.6f}",
                "hist_distance": f"{item.hist_distance:.6f}",
                "center_distance": f"{item.center_distance:.6f}",
                "track_median_hist_distance": f"{median_hist:.6f}",
                "track_p90_match_cost": f"{p90_cost:.6f}",
                "track_median_frame_gap": f"{statistics.median(frame_gaps) if frame_gaps else 0:.3f}",
            })

    fields = list(output_rows[0]) if output_rows else []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "schema_version": "bmd45-temporal-tracks-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if valid_tracks else "fail",
        "coco": str(args.coco.resolve()),
        "coco_sha256": sha256(args.coco),
        "crop_root": str(args.crop_root.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": sha256(args.output_csv),
        "source_license": "CC-BY-4.0",
        "split_used": "train",
        "sequences": len(sequence_images),
        "mapped_detections_seen": detections_seen,
        "tracks_total": len(finished_tracks),
        "valid_tracks": valid_tracks,
        "valid_rows": valid_rows,
        "valid_track_type_counts": dict(sorted(valid_type_counts.items())),
        "track_length_histogram": {str(key): value for key, value in sorted(length_histogram.items())},
        "valid_track_median_hist_distance_p50": percentile(track_median_hist_distances, 0.50),
        "valid_track_p90_match_cost_p90": percentile(track_p90_match_costs, 0.90),
        "skipped": dict(sorted(skipped.items())),
        "missing_crop_examples": missing_crops,
        "parameters": {
            "max_frame_gap": args.max_frame_gap,
            "max_match_cost": args.max_match_cost,
            "minimum_track_frames": args.minimum_track_frames,
            "maximum_track_median_hist_distance": args.maximum_track_median_hist_distance,
            "maximum_track_p90_match_cost": args.maximum_track_p90_match_cost,
        },
        "policy": {
            "individual_vehicle_tracks_not_frame_groups": True,
            "source_category_motion_geometry_and_hsv_appearance_required": True,
            "color_labels_created": False,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only track_valid=true rows may enter the Stage52 train-only color pseudo-label audit",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "mapped_detections_seen": detections_seen,
        "tracks_total": len(finished_tracks),
        "valid_tracks": valid_tracks,
        "valid_rows": valid_rows,
        "output_csv_sha256": report["output_csv_sha256"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
