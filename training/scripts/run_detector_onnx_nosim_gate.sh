#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT=/root/autodl-tmp/vcas
CODE_ROOT="$VCAS_ROOT/code-20260811-domain-v1"
RUN_ROOT="$VCAS_ROOT/runs/detection/DET-DOMAIN-BMD45-FINETUNE-R1"
EVAL_ROOT="$VCAS_ROOT/manifests/domain-retrain-v1/evaluation"
PT_PATH="$RUN_ROOT/weights/best.pt"
ONNX_PATH="$RUN_ROOT/weights/best-nosim-op17.onnx"

cd "$CODE_ROOT"
/root/miniconda3/bin/python - "$PT_PATH" <<'PY'
import sys
from ultralytics import YOLO

YOLO(sys.argv[1]).export(
    format="onnx",
    imgsz=960,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
    device=0,
)
PY
mv "$RUN_ROOT/weights/best.onnx" "$ONNX_PATH"

/root/miniconda3/bin/python training/scripts/evaluate_detector_domain.py \
  --model "$ONNX_PATH" \
  --data "$VCAS_ROOT/datasets/dataset-domain-bmd45-v1/detection/vehicle_det_v1.yaml" \
  --split test --imgsz 960 --batch 1 --device 0 \
  --project "$VCAS_ROOT/runs/detection" \
  --name DET-DOMAIN-BMD45-FINETUNE-R1-ONNX-NOSIM-ORIGINAL \
  --output "$EVAL_ROOT/onnx-nosim-original-test.json"

/root/miniconda3/bin/python training/scripts/evaluate_detector_domain.py \
  --model "$ONNX_PATH" \
  --data "$VCAS_ROOT/datasets/dataset-domain-bmd45-v1/detection/bmd45_val.yaml" \
  --split test --imgsz 960 --batch 1 --device 0 \
  --project "$VCAS_ROOT/runs/detection" \
  --name DET-DOMAIN-BMD45-FINETUNE-R1-ONNX-NOSIM-BMD45 \
  --output "$EVAL_ROOT/onnx-nosim-bmd45.json"
