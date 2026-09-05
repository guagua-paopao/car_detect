#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
OUT=$BASE/runs/stage20/ATTR-V17-CONVNEXT-TINY-256-HARD
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest $BASE/datasets/attribute-domain-v2/attribute_manifest.hard-v2.csv \
  --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture convnext_tiny --resize-mode stretch \
  --epochs 6 --batch-size 32 --workers 8 --learning-rate 1e-4 --weight-decay 1e-4 \
  --color-loss-weight 0.45 --body-loss-weight 1.25 --focal-gamma 0.35 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 1.0 --small-sample-weight 1.0 \
  --freeze-backbone-epochs 1 --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-convnext-hard-v20 --code-revision stage20-convnext-hard-v20 \
  --output-dir "$OUT"
