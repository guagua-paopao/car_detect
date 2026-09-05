"""Overlay deployed vehicle body-type and color predictions on detector video results."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--type-threshold", type=float, default=0.75)
    parser.add_argument("--color-threshold", type=float, default=0.70)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import onnxruntime as ort

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    body_types = list(labels["body_types"])
    colors = list(labels["colors"])
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output: {args.output_video}")

    detection_rows = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(detection_rows) != frame_count:
        raise RuntimeError(f"detections {len(detection_rows)} != video frames {frame_count}")
    body_counts = Counter()
    color_counts = Counter()
    window_body = Counter()
    window_color = Counter()
    total_crops = 0
    window_crops = 0
    inference_seconds = 0.0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detection_rows[frame_index]["detections"]
        crops: list[np.ndarray] = []
        valid_indices: list[int] = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
            x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (args.input_size, args.input_size), interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crops.append(np.transpose(crop, (2, 0, 1)))
            valid_indices.append(index)
        predictions: dict[int, tuple[str, float, str, float]] = {}
        started = time.perf_counter()
        for start in range(0, len(crops), args.batch):
            batch = np.asarray(crops[start : start + args.batch], dtype=np.float32)
            if not len(batch):
                continue
            body_logits, color_logits = session.run(None, {input_name: batch})
            body_probs = softmax(np.asarray(body_logits))
            color_probs = softmax(np.asarray(color_logits))
            for offset in range(len(batch)):
                body_id = int(body_probs[offset].argmax())
                color_id = int(color_probs[offset].argmax())
                body_conf = float(body_probs[offset, body_id])
                color_conf = float(color_probs[offset, color_id])
                body_label = body_types[body_id] if body_conf >= args.type_threshold else "unknown"
                color_label = colors[color_id] if color_conf >= args.color_threshold else "unknown"
                predictions[valid_indices[start + offset]] = (body_label, body_conf, color_label, color_conf)
        inference_seconds += time.perf_counter() - started
        total_crops += len(predictions)
        in_window = 36.0 <= frame_index / fps < 48.0
        if in_window:
            window_crops += len(predictions)
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
            x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
            y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
            body_label, body_conf, color_label, color_conf = predictions.get(index, ("unknown", 0.0, "unknown", 0.0))
            body_counts[body_label] += 1
            color_counts[color_label] += 1
            if in_window:
                window_body[body_label] += 1
                window_color[color_label] += 1
            text = f"{detection['class_name']} {detection['confidence']:.2f} | {body_label} {body_conf:.2f} | {color_label} {color_conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
            text_y = max(18, y1 - 6)
            cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"NEW DETECTOR + ATTRIBUTES | {frame_index / fps:05.1f}s", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    report = {
        "schema_version": "1.0",
        "video": {"path": str(args.video), "fps": fps, "frames": frame_index, "width": width, "height": height},
        "model": str(args.model),
        "settings": {"batch": args.batch, "input_size": args.input_size, "type_threshold": args.type_threshold, "color_threshold": args.color_threshold, "providers": session.get_providers()},
        "runtime": {"attribute_crops": total_crops, "inference_seconds": inference_seconds, "ms_per_crop": inference_seconds * 1000.0 / max(1, total_crops)},
        "all_video": {"body_type": dict(body_counts), "color": dict(color_counts)},
        "window_36_48s": {"attribute_crops": window_crops, "body_type": dict(window_body), "color": dict(window_color)},
        "output_video": str(args.output_video),
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
