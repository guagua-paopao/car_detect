#!/usr/bin/env python3
"""Validate detector PT/ONNX parity and benchmark ONNX CUDA/TensorRT providers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--match-iou", type=float, default=0.90)
    return parser.parse_args()


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def load_frames(video: Path, count: int) -> tuple[list[np.ndarray], list[int]]:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted({round(value) for value in np.linspace(0, frame_count - 1, count)})
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index}")
        frames.append(frame)
    capture.release()
    return frames, indices


def detections(results: list) -> list[list[dict]]:
    output: list[list[dict]] = []
    for result in results:
        rows: list[dict] = []
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in box],
                    }
                )
        output.append(rows)
    return output


def compare(reference: list[list[dict]], candidate: list[list[dict]], match_iou: float) -> dict:
    matched = 0
    reference_only = 0
    candidate_only = 0
    confidence_deltas: list[float] = []
    count_deltas: list[int] = []
    for ref_rows, candidate_rows in zip(reference, candidate, strict=True):
        used: set[int] = set()
        frame_matched = 0
        for ref in ref_rows:
            best: tuple[float, int] | None = None
            for index, row in enumerate(candidate_rows):
                if index in used or row["class_id"] != ref["class_id"]:
                    continue
                overlap = iou(ref["xyxy"], row["xyxy"])
                if overlap >= match_iou and (best is None or overlap > best[0]):
                    best = (overlap, index)
            if best is None:
                reference_only += 1
            else:
                used.add(best[1])
                frame_matched += 1
                confidence_deltas.append(abs(ref["confidence"] - candidate_rows[best[1]]["confidence"]))
        matched += frame_matched
        candidate_only += len(candidate_rows) - len(used)
        count_deltas.append(abs(len(ref_rows) - len(candidate_rows)))
    denominator = matched + reference_only + candidate_only
    return {
        "matched": matched,
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "match_rate": matched / denominator if denominator else 1.0,
        "mean_confidence_delta": mean(confidence_deltas) if confidence_deltas else 0.0,
        "max_confidence_delta": max(confidence_deltas, default=0.0),
        "max_count_delta_per_frame": max(count_deltas, default=0),
    }


def preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    from ultralytics.data.augment import LetterBox

    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=frame)
    image = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(image, dtype=np.float32)[None] / 255.0


def benchmark_session(session, input_name: str, tensor: np.ndarray, iterations: int) -> dict:
    for _ in range(10):
        session.run(None, {input_name: tensor})
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, {input_name: tensor})
        durations.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(durations)
    return {
        "mean_ms": mean(durations),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "iterations": iterations,
    }


def main() -> int:
    args = parse_args()
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    frames, indices = load_frames(args.video.resolve(), args.samples)
    pt_model = YOLO(str(args.pt.resolve()))
    onnx_model = YOLO(str(args.onnx.resolve()), task="detect")
    pt_results = pt_model.predict(
        source=frames, imgsz=args.imgsz, conf=args.confidence, iou=0.70,
        device=args.device, half=False, verbose=False,
    )
    onnx_results = []
    for frame in frames:
        # The released detector contract is fixed batch=1.
        onnx_results.extend(
            onnx_model.predict(
                source=frame, imgsz=args.imgsz, conf=args.confidence, iou=0.70,
                device=args.device, half=False, batch=1, verbose=False,
            )
        )
    parity = compare(detections(pt_results), detections(onnx_results), args.match_iou)

    tensor = preprocess(frames[len(frames) // 2], args.imgsz)
    torch_model = pt_model.model.eval().to("cuda")
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(tensor).to("cuda"))
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    torch_array = torch_output.detach().cpu().numpy()

    cuda_session = ort.InferenceSession(
        str(args.onnx.resolve()), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = cuda_session.get_inputs()[0].name
    cuda_output = cuda_session.run(None, {input_name: tensor})[0]
    raw_delta = np.abs(torch_array.astype(np.float32) - cuda_output.astype(np.float32))
    cuda_benchmark = benchmark_session(
        cuda_session, input_name, tensor, args.benchmark_iterations
    )

    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    trt_options = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(args.cache_dir.resolve()),
        "trt_timing_cache_enable": True,
    }
    trt_session = ort.InferenceSession(
        str(args.onnx.resolve()),
        providers=[
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    trt_benchmark = benchmark_session(
        trt_session, trt_session.get_inputs()[0].name, tensor, args.benchmark_iterations
    )
    trt_output = trt_session.run(None, {trt_session.get_inputs()[0].name: tensor})[0]
    trt_delta = np.abs(cuda_output.astype(np.float32) - trt_output.astype(np.float32))

    passed = (
        parity["match_rate"] >= 0.98
        and parity["mean_confidence_delta"] <= 0.01
        and float(raw_delta.mean()) <= 0.01
        and float(trt_delta.mean()) <= 0.01
        and "TensorrtExecutionProvider" in trt_session.get_providers()
    )
    payload = {
        "schema_version": "1.0",
        "status": "pass" if passed else "fail",
        "video": str(args.video.resolve()),
        "sample_frame_indices": indices,
        "confidence": args.confidence,
        "postprocess_parity": parity,
        "raw_pt_vs_onnx_cuda": {
            "mean_absolute_delta": float(raw_delta.mean()),
            "p99_absolute_delta": float(np.quantile(raw_delta, 0.99)),
            "max_absolute_delta": float(raw_delta.max()),
        },
        "raw_onnx_cuda_vs_tensorrt": {
            "mean_absolute_delta": float(trt_delta.mean()),
            "p99_absolute_delta": float(np.quantile(trt_delta, 0.99)),
            "max_absolute_delta": float(trt_delta.max()),
        },
        "onnx_cuda_benchmark": cuda_benchmark,
        "onnx_tensorrt_benchmark": trt_benchmark,
        "onnx_providers": ort.get_available_providers(),
        "tensorrt_session_providers": trt_session.get_providers(),
        "portability_note": "TensorRT cache is server-specific; rebuild the production engine on the Windows target host.",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
