#!/usr/bin/env python3
"""Prepare frozen-video tensors and ONNX Runtime references for TensorRT parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = size - resized_width
    dh = size - resized_height
    left = round(dw / 2 - 0.1)
    right = round(dw / 2 + 0.1)
    top = round(dh / 2 - 0.1)
    bottom = round(dh / 2 + 0.1)
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", nargs="+", type=float, default=[36.0, 40.0, 44.0, 47.5])
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    records = []
    for index, second in enumerate(args.seconds):
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame at {second:.3f}s")
        tensor = letterbox(frame, args.imgsz)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        reference = session.run([output_name], {input_name: tensor})[0]
        stem = f"frame_{index:02d}_{second:06.2f}s"
        raw_path = args.output / f"{stem}.raw"
        reference_path = args.output / f"{stem}.onnx.npy"
        tensor.tofile(raw_path)
        np.save(reference_path, reference)
        records.append(
            {
                "second": second,
                "input_raw": raw_path.name,
                "onnx_reference": reference_path.name,
                "input_shape": list(tensor.shape),
                "output_shape": list(reference.shape),
                "output_min": float(reference.min()),
                "output_max": float(reference.max()),
            }
        )
    capture.release()
    manifest = {
        "video": str(args.video.resolve()),
        "onnx": str(args.onnx.resolve()),
        "input_name": input_name,
        "output_name": output_name,
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
