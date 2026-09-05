#!/usr/bin/env python3
"""Compare TensorRT FP16 exports with ONNX Runtime on identical detector inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detections(output: np.ndarray, confidence: float, iou: float) -> list[dict]:
    values = output[0]
    boxes_xywh = values[:4].T
    scores_by_class = values[4:].T
    classes = scores_by_class.argmax(axis=1)
    scores = scores_by_class[np.arange(len(classes)), classes]
    keep = scores >= confidence
    boxes_xywh = boxes_xywh[keep]
    classes = classes[keep]
    scores = scores[keep]
    result: list[dict] = []
    for class_id in np.unique(classes):
        class_mask = classes == class_id
        class_boxes = boxes_xywh[class_mask]
        class_scores = scores[class_mask]
        xyxy = np.empty_like(class_boxes)
        xyxy[:, 0] = class_boxes[:, 0] - class_boxes[:, 2] / 2
        xyxy[:, 1] = class_boxes[:, 1] - class_boxes[:, 3] / 2
        xyxy[:, 2] = class_boxes[:, 0] + class_boxes[:, 2] / 2
        xyxy[:, 3] = class_boxes[:, 1] + class_boxes[:, 3] / 2
        nms_boxes = np.column_stack((xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2])).tolist()
        selected = cv2.dnn.NMSBoxes(nms_boxes, class_scores.tolist(), confidence, iou)
        for local_index in np.asarray(selected).reshape(-1):
            result.append(
                {
                    "class_id": int(class_id),
                    "score": float(class_scores[local_index]),
                    "box": xyxy[local_index].astype(float).tolist(),
                }
            )
    return sorted(result, key=lambda item: item["score"], reverse=True)


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def match(
    reference: list[dict],
    candidate: list[dict],
    reference_near_threshold: list[dict],
    candidate_near_threshold: list[dict],
) -> dict:
    available = set(range(len(candidate)))
    pairs = []
    for ref_index, ref in enumerate(reference):
        options = [
            (box_iou(ref["box"], candidate[idx]["box"]), idx)
            for idx in available
            if candidate[idx]["class_id"] == ref["class_id"]
        ]
        if not options:
            continue
        overlap, candidate_index = max(options)
        if overlap < 0.5:
            continue
        available.remove(candidate_index)
        pairs.append(
            {
                "reference_index": ref_index,
                "candidate_index": candidate_index,
                "iou": overlap,
                "score_absolute_delta": abs(ref["score"] - candidate[candidate_index]["score"]),
            }
        )
    unmatched_reference = [index for index in range(len(reference)) if index not in {pair["reference_index"] for pair in pairs}]
    unmatched_candidate = sorted(available)

    def has_boundary_equivalent(item: dict, pool: list[dict]) -> bool:
        return any(
            other["class_id"] == item["class_id"] and box_iou(item["box"], other["box"]) >= 0.95
            for other in pool
        )

    boundary_reference = sum(
        has_boundary_equivalent(reference[index], candidate_near_threshold)
        for index in unmatched_reference
    )
    boundary_candidate = sum(
        has_boundary_equivalent(candidate[index], reference_near_threshold)
        for index in unmatched_candidate
    )
    return {
        "reference_count": len(reference),
        "tensorrt_count": len(candidate),
        "matched_count": len(pairs),
        "reference_unmatched": len(reference) - len(pairs),
        "tensorrt_unmatched": len(candidate) - len(pairs),
        "reference_threshold_boundary_equivalents": boundary_reference,
        "tensorrt_threshold_boundary_equivalents": boundary_candidate,
        "reference_true_unmatched": len(unmatched_reference) - boundary_reference,
        "tensorrt_true_unmatched": len(unmatched_candidate) - boundary_candidate,
        "minimum_matched_iou": min((pair["iou"] for pair in pairs), default=None),
        "mean_matched_iou": float(np.mean([pair["iou"] for pair in pairs])) if pairs else None,
        "maximum_score_absolute_delta": max((pair["score_absolute_delta"] for pair in pairs), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        stem = Path(record["onnx_reference"]).name.removesuffix(".onnx.npy")
        onnx = np.load(args.directory / record["onnx_reference"])
        trt_document = json.loads((args.directory / f"{stem}.trt.json").read_text(encoding="utf-8"))
        trt = np.asarray(trt_document[0]["values"], dtype=np.float32).reshape(onnx.shape)
        difference = np.abs(onnx - trt)
        onnx_detections = detections(onnx, args.confidence, args.iou)
        trt_detections = detections(trt, args.confidence, args.iou)
        boundary_confidence = max(0.0, args.confidence - 0.01)
        onnx_near_threshold = detections(onnx, boundary_confidence, args.iou)
        trt_near_threshold = detections(trt, boundary_confidence, args.iou)
        records.append(
            {
                "second": record["second"],
                "raw": {
                    "mean_absolute_delta": float(difference.mean()),
                    "p99_absolute_delta": float(np.quantile(difference, 0.99)),
                    "maximum_absolute_delta": float(difference.max()),
                    "class_score_mean_absolute_delta": float(difference[:, 4:, :].mean()),
                    "class_score_maximum_absolute_delta": float(difference[:, 4:, :].max()),
                },
                "postprocess": match(
                    onnx_detections,
                    trt_detections,
                    onnx_near_threshold,
                    trt_near_threshold,
                ),
            }
        )
    all_matched_or_boundary_equivalent = all(
        record["postprocess"]["reference_true_unmatched"] == 0
        and record["postprocess"]["tensorrt_true_unmatched"] == 0
        for record in records
    )
    minimum_iou = min(record["postprocess"]["minimum_matched_iou"] for record in records)
    max_matched_score_delta = max(
        record["postprocess"]["maximum_score_absolute_delta"] for record in records
    )
    class_score_mean_delta = max(
        record["raw"]["class_score_mean_absolute_delta"] for record in records
    )
    passed = (
        all_matched_or_boundary_equivalent
        and minimum_iou >= 0.95
        and max_matched_score_delta <= 0.005
        and class_score_mean_delta <= 0.0001
    )
    report = {
        "status": "pass" if passed else "fail",
        "gate": {
            "all_detections_matched_or_within_0_01_confidence_boundary": all_matched_or_boundary_equivalent,
            "minimum_matched_iou_gte_0_95": minimum_iou >= 0.95,
            "maximum_matched_score_absolute_delta_lte_0_005": max_matched_score_delta <= 0.005,
            "maximum_frame_class_score_mean_absolute_delta_lte_0_0001": class_score_mean_delta <= 0.0001,
        },
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
