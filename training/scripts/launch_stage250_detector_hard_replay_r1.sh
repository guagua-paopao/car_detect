#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/root/autodl-tmp/vcas/code-stage250-detector
DATA_ROOT=/root/autodl-tmp/vcas/datasets/det-stage250-hard-replay-r1
RUN_ROOT=/root/autodl-tmp/vcas/runs/detection-stage250
BASE_DATA=/root/autodl-tmp/vcas/datasets/dataset-large-v1-sixclass-ontology-r5/detection
BMD_DATA=/root/autodl-tmp/vcas/datasets/dataset-domain-bmd45-v1/detection
BASE_MODEL=/root/autodl-tmp/vcas/runs/detection/DET-DOMAIN-BMD45-FINETUNE-R1/weights/best.pt
CANDIDATE=DET-STAGE250-HARD-REPLAY-960-R1
PYTHON=/root/miniconda3/bin/python

mkdir -p "$DATA_ROOT" "$RUN_ROOT/evaluation"

"$PYTHON" "$CODE_ROOT/build_stage250_detector_manifests.py" \
  --base-root "$BASE_DATA" --bmd-root "$BMD_DATA" --output-root "$DATA_ROOT"

"$PYTHON" "$CODE_ROOT/evaluate_detector_validation_only.py" \
  --model "$BASE_MODEL" --data "$DATA_ROOT/base-validation.yaml" --imgsz 960 --batch 16 \
  --project "$RUN_ROOT/evaluation" --name baseline-base-val \
  --output "$RUN_ROOT/evaluation/baseline-base-val.json"

"$PYTHON" "$CODE_ROOT/evaluate_detector_validation_only.py" \
  --model "$BASE_MODEL" --data "$DATA_ROOT/bmd45-validation.yaml" --imgsz 960 --batch 16 \
  --project "$RUN_ROOT/evaluation" --name baseline-bmd45-val \
  --output "$RUN_ROOT/evaluation/baseline-bmd45-val.json"

"$PYTHON" "$CODE_ROOT/train_detector_validation_only.py" \
  --data "$DATA_ROOT/train-balanced.yaml" --model "$BASE_MODEL" --imgsz 960 --epochs 24 \
  --batch -1 --device 0 --workers 8 --patience 8 --optimizer AdamW --lr0 0.0003 \
  --lrf 0.05 --weight-decay 0.0005 --warmup-epochs 1 --close-mosaic 6 --seed 20260903 \
  --project "$RUN_ROOT" --name "$CANDIDATE" --dataset-version det-stage250-hard-replay-r1

CANDIDATE_MODEL="$RUN_ROOT/$CANDIDATE/weights/best.pt"
"$PYTHON" "$CODE_ROOT/evaluate_detector_validation_only.py" \
  --model "$CANDIDATE_MODEL" --data "$DATA_ROOT/base-validation.yaml" --imgsz 960 --batch 16 \
  --project "$RUN_ROOT/evaluation" --name candidate-base-val \
  --output "$RUN_ROOT/evaluation/candidate-base-val.json"

"$PYTHON" "$CODE_ROOT/evaluate_detector_validation_only.py" \
  --model "$CANDIDATE_MODEL" --data "$DATA_ROOT/bmd45-validation.yaml" --imgsz 960 --batch 16 \
  --project "$RUN_ROOT/evaluation" --name candidate-bmd45-val \
  --output "$RUN_ROOT/evaluation/candidate-bmd45-val.json"

"$PYTHON" "$CODE_ROOT/finalize_stage250_detector_gate.py" \
  --baseline-base "$RUN_ROOT/evaluation/baseline-base-val.json" \
  --baseline-bmd "$RUN_ROOT/evaluation/baseline-bmd45-val.json" \
  --candidate-base "$RUN_ROOT/evaluation/candidate-base-val.json" \
  --candidate-bmd "$RUN_ROOT/evaluation/candidate-bmd45-val.json" \
  --output "$RUN_ROOT/stage250-validation-gate.json"
