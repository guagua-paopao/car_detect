#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh26-hard-v3.csv
OUT=$BASE/runs/stage6/ATTR-V4-MNV3-256-DUAL-UVH26
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large_dual --resize-mode stretch \
  --epochs 12 --batch-size 64 --workers 8 --learning-rate 2e-4 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 1.2 --focal-gamma 0.8 --color-focal-gamma 0.2 \
  --class-weighting inverse_sqrt --label-smoothing 0.04 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 1.5 \
  --teacher-checkpoint $BASE/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.10 --distill-temperature 2.5 --run-kind formal \
  --dataset-version attribute-domain-v2-uvh26-dual-v5 --code-revision stage6-dual-v5 \
  --output-dir "$OUT"
