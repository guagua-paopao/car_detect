#!/usr/bin/env python3
"""Prepare validation-only raw tensors and ONNX references for Stage71 TRT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {label} SHA256 mismatch: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} eligible rows; need {count}")
    if count == 1:
        return [rows[0]]
    indexes = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indexes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--expected-export-report-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite TensorRT parity inputs")
    if args.batch_size < 8:
        raise RuntimeError("TensorRT parity batch must contain at least 8 samples")
    require_sha(args.export_report, args.expected_export_report_sha256, "export report")
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    export_report = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export_report.get("status") != "pass_onnx_exported_candidate_only":
        raise RuntimeError("export report is not TensorRT-parity eligible")
    if export_report.get("policy", {}).get("frozen_video_used") is not False:
        raise RuntimeError("export report used frozen video")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is unavailable") from exc

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == args.split
            and (row.get("review_status", "approved") or "approved") == "approved"
        ]
    selected = evenly_spaced(rows, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models: dict[str, Any] = {}
    for head in ("body", "color"):
        evidence = export_report["exports"][head]
        onnx_path = Path(evidence["onnx"])
        require_sha(onnx_path, evidence["onnx_sha256"], f"{head} ONNX")
        input_size = int(evidence["input_size"])
        raw_images = []
        for row in selected:
            with Image.open(args.manifest.parent / row["image_path"]) as image:
                resized = image.convert("RGB").resize(
                    (input_size, input_size),
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(resized, dtype=np.float32) / 255.0
            raw_images.append(np.transpose(array, (2, 0, 1)))
        raw = np.ascontiguousarray(raw_images, dtype=np.float32)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        if session.get_inputs()[0].name != "images":
            raise RuntimeError(f"unexpected {head} ONNX input")
        output_names = [item.name for item in session.get_outputs()]
        if output_names != ["body_type", "color"]:
            raise RuntimeError(f"unexpected {head} ONNX outputs: {output_names}")
        body_logits, color_logits = session.run(None, {"images": raw})
        raw_path = args.output_dir / f"{head}-input.raw"
        body_path = args.output_dir / f"{head}-body_type.onnx.npy"
        color_path = args.output_dir / f"{head}-color.onnx.npy"
        raw.tofile(raw_path)
        np.save(body_path, body_logits)
        np.save(color_path, color_logits)
        models[head] = {
            "routed_output": "body_type" if head == "body" else "color",
            "onnx": str(onnx_path.resolve()),
            "onnx_sha256": sha256(onnx_path),
            "input_size": input_size,
            "batch_size": args.batch_size,
            "input_shape": list(raw.shape),
            "input_raw": raw_path.name,
            "input_raw_sha256": sha256(raw_path),
            "references": {
                "body_type": {
                    "path": body_path.name,
                    "sha256": sha256(body_path),
                    "shape": list(body_logits.shape),
                },
                "color": {
                    "path": color_path.name,
                    "sha256": sha256(color_path),
                    "shape": list(color_logits.shape),
                },
            },
        }

    require_sha(args.export_report, args.expected_export_report_sha256, "export report")
    require_sha(args.manifest, args.expected_manifest_sha256, "validation manifest")
    report = {
        "schema_version": "stage71-attribute-trt-parity-inputs-v1",
        "status": "pass_inputs_prepared",
        "split": args.split,
        "batch_size": args.batch_size,
        "selection": "deterministic evenly spaced approved validation rows",
        "selected_rows": [
            {
                "image_path": row["image_path"],
                "source_dataset": row.get("source_dataset"),
                "camera_id": row.get("camera_id"),
                "video_id": row.get("video_id"),
                "track_id": row.get("track_id"),
            }
            for row in selected
        ],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "export_report": str(args.export_report.resolve()),
        "export_report_sha256": sha256(args.export_report),
        "models": models,
        "policy": {
            "validation_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, report)
    print(json.dumps({"status": report["status"], "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
