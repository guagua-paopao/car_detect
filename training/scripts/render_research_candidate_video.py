#!/usr/bin/env python3
"""Render a non-frozen research-candidate vehicle attribute video preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from src.multitask_mobilenet_v3 import model_from_checkpoint


FORBIDDEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, scale, left, top


def detector_predictions(
    session: ort.InferenceSession,
    frame: np.ndarray,
    input_size: int,
    confidence: float,
    nms_iou: float,
) -> list[tuple[list[int], float, int]]:
    height, width = frame.shape[:2]
    padded, scale, left, top = letterbox(frame, input_size)
    tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].T
    boxes: list[list[int]] = []
    scores: list[float] = []
    classes: list[int] = []
    for row in raw:
        class_id = int(np.argmax(row[4:]))
        score = float(row[4 + class_id])
        if score < confidence:
            continue
        center_x, center_y, box_width, box_height = map(float, row[:4])
        x1 = int(max(0, min(width - 1, (center_x - box_width / 2 - left) / scale)))
        y1 = int(max(0, min(height - 1, (center_y - box_height / 2 - top) / scale)))
        x2 = int(max(0, min(width, (center_x + box_width / 2 - left) / scale)))
        y2 = int(max(0, min(height, (center_y + box_height / 2 - top) / scale)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
        classes.append(class_id)
    keep = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_iou)
    if len(keep) == 0:
        return []
    return [(boxes[int(index)], scores[int(index)], classes[int(index)]) for index in np.asarray(keep).reshape(-1)]


def iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def load_attribute(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = model_from_checkpoint(checkpoint, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)
    return model, checkpoint


def preprocess(crops: list[np.ndarray], checkpoint: dict, device: torch.device) -> torch.Tensor:
    size = int(checkpoint["input_size"])
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    rows = []
    for crop in crops:
        image = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - mean) / std
        rows.append(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(np.stack(rows)).to(device)


def class_thresholds(report_path: Path) -> dict[str, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresholds = report.get("selection", {}).get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        thresholds = report.get("selected_color", {}).get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RuntimeError(f"missing validation-selected class thresholds: {report_path}")
    return {str(key): float(value) for key, value in thresholds.items()}


def fuse(history: deque[tuple[np.ndarray, float]]) -> np.ndarray:
    weights = np.asarray([quality for _, quality in history], dtype=np.float32)
    matrix = np.stack([probability for probability, _ in history])
    return np.average(matrix, axis=0, weights=np.maximum(weights, 1e-6))


def select_label(probability: np.ndarray, labels: list[str], thresholds: dict[str, float]) -> tuple[str, float]:
    class_id = int(np.argmax(probability))
    label = labels[class_id]
    confidence = float(probability[class_id])
    if label == "unknown" or confidence < thresholds.get(label, 1.01):
        return "unknown", confidence
    return label, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--body-checkpoint", type=Path, required=True)
    parser.add_argument("--body-report", type=Path, required=True)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--color-report", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.40)
    parser.add_argument("--detector-iou", type=float, default=0.50)
    parser.add_argument("--history", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    searchable = " ".join(str(value).lower() for value in (args.video, args.output_video, args.output_report))
    if any(marker in searchable for marker in FORBIDDEN_MARKERS):
        raise RuntimeError("frozen video or frozen-window marker is forbidden in research preview")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    vehicle_classes = list(labels["vehicle_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    detector = ort.InferenceSession(str(args.detector), providers=detector_providers)
    detector_shape = detector.get_inputs()[0].shape
    detector_size = int(detector_shape[-1]) if isinstance(detector_shape[-1], int) else 960
    body_model, body_checkpoint = load_attribute(args.body_checkpoint, device)
    color_model, color_checkpoint = load_attribute(args.color_checkpoint, device)
    body_labels = list(body_checkpoint["body_types"])
    color_labels = list(color_checkpoint["colors"])
    if body_labels != list(labels["body_types"]) or color_labels != list(labels["colors"]):
        raise RuntimeError("candidate label contract mismatch")
    body_threshold = class_thresholds(args.body_report)
    color_threshold = class_thresholds(args.color_report)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {args.output_video}")

    tracks: dict[int, dict] = {}
    next_track = 1
    frame_index = 0
    observations = 0
    zero_detection_frames = 0
    body_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    body_switches = 0
    color_switches = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = detector_predictions(detector, frame, detector_size, args.detector_confidence, args.detector_iou)
        if not detections:
            zero_detection_frames += 1
        assignments = []
        used: set[int] = set()
        crops: list[np.ndarray] = []
        valid_rows = []
        for detection_index, (box, score, class_id) in enumerate(detections):
            best_track, best_iou = None, 0.0
            for track_id, track in tracks.items():
                if track_id in used or frame_index - track["last_frame"] > 10:
                    continue
                overlap = iou(box, track["box"])
                if overlap > best_iou:
                    best_track, best_iou = track_id, overlap
            if best_track is None or best_iou < 0.25:
                best_track = next_track
                next_track += 1
                tracks[best_track] = {
                    "box": box,
                    "last_frame": frame_index,
                    "body": deque(maxlen=args.history),
                    "color": deque(maxlen=args.history),
                    "last_body": "unknown",
                    "last_color": "unknown",
                }
            used.add(best_track)
            tracks[best_track]["box"] = box
            tracks[best_track]["last_frame"] = frame_index
            assignments.append((detection_index, best_track, box, score, class_id))
            x, y, box_width, box_height = box
            crop = frame[y : y + box_height, x : x + box_width]
            if crop.size:
                crops.append(crop)
                valid_rows.append((detection_index, best_track, box, score, class_id))

        prediction_by_detection = {}
        if crops:
            with torch.inference_mode():
                body_logits, _ = body_model(preprocess(crops, body_checkpoint, device))
                _, color_logits = color_model(preprocess(crops, color_checkpoint, device))
                body_probabilities = torch.softmax(body_logits, dim=1).cpu().numpy()
                color_probabilities = torch.softmax(color_logits, dim=1).cpu().numpy()
            for row, body_probability, color_probability in zip(valid_rows, body_probabilities, color_probabilities):
                detection_index, track_id, box, score, class_id = row
                area_fraction = (box[2] * box[3]) / max(1, width * height)
                quality = max(1e-4, float(score) * math.sqrt(max(area_fraction, 1e-8)))
                track = tracks[track_id]
                track["body"].append((body_probability, quality))
                track["color"].append((color_probability, quality))
                body_label, body_confidence = select_label(fuse(track["body"]), body_labels, body_threshold)
                color_label, color_confidence = select_label(fuse(track["color"]), color_labels, color_threshold)
                if track["last_body"] not in {"unknown", body_label} and body_label != "unknown":
                    body_switches += 1
                if track["last_color"] not in {"unknown", color_label} and color_label != "unknown":
                    color_switches += 1
                if body_label != "unknown":
                    track["last_body"] = body_label
                if color_label != "unknown":
                    track["last_color"] = color_label
                prediction_by_detection[detection_index] = (track_id, body_label, body_confidence, color_label, color_confidence)
                body_counts[body_label] += 1
                color_counts[color_label] += 1
                observations += 1

        for detection_index, (_, score, class_id) in enumerate(detections):
            box = detections[detection_index][0]
            x, y, box_width, box_height = box
            track_id, body_label, body_confidence, color_label, color_confidence = prediction_by_detection.get(
                detection_index, (0, "unknown", 0.0, "unknown", 0.0)
            )
            class_name = vehicle_classes[class_id] if class_id < len(vehicle_classes) else "vehicle"
            text = f"T{track_id} {class_name} {score:.2f} | {body_label} {body_confidence:.2f} | {color_label} {color_confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 215, 255), 2)
            text_y = max(18, y - 6)
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, 56), (0, 0, 0), -1)
        cv2.putText(frame, "LATEST RESEARCH CANDIDATE | NON-FROZEN VIDEO | NOT DEPLOYED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"type + color | 5-frame quality fusion | {frame_index / fps:05.1f}s", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 255, 190), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    report = {
        "schema_version": "research-candidate-video-preview-v1",
        "status": "complete_behavior_preview_no_ground_truth",
        "video": {"path": str(args.video), "sha256": sha256(args.video), "frames": frame_index, "reported_frames": total_frames, "fps": fps, "width": width, "height": height},
        "models": {
            "detector": {"path": str(args.detector), "sha256": sha256(args.detector), "providers": detector.get_providers()},
            "body": {"path": str(args.body_checkpoint), "sha256": sha256(args.body_checkpoint), "validation_report": str(args.body_report), "validation_report_sha256": sha256(args.body_report)},
            "color": {"path": str(args.color_checkpoint), "sha256": sha256(args.color_checkpoint), "validation_report": str(args.color_report), "validation_report_sha256": sha256(args.color_report)},
        },
        "settings": {"detector_confidence": args.detector_confidence, "detector_iou": args.detector_iou, "history": args.history, "body_thresholds": body_threshold, "color_thresholds": color_threshold},
        "results": {
            "observations": observations,
            "tracks_created": next_track - 1,
            "zero_detection_frames": zero_detection_frames,
            "body_counts": dict(body_counts),
            "color_counts": dict(color_counts),
            "body_unknown_rate": body_counts["unknown"] / max(1, observations),
            "color_unknown_rate": color_counts["unknown"] / max(1, observations),
            "body_label_switches": body_switches,
            "color_label_switches": color_switches,
        },
        "ground_truth": "unavailable; visual behavior demonstration only, not an accuracy measurement",
        "research_only": True,
        "deployment_eligible": False,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "output_video": str(args.output_video),
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(f"{sha256(args.output_report)}  {args.output_report.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_video": str(args.output_video), "results": report["results"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
