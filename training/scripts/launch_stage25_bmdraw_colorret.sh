#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v23.csv
OUT=$BASE/runs/stage25/ATTR-V25-MNV3-256-BMDRAW-COLORRET
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 4 --batch-size 64 --workers 8 --learning-rate 1.5e-5 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 0.30 --focal-gamma 0.15 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.25 --small-sample-weight 0.50 --color-sample-weight 8.0 \
  --init-checkpoint $BASE/runs/stage23/ATTR-V23-MNV3-256-BMDRAW-TYPE/best.pt \
  --freeze-backbone-epochs 1 --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-bmdraw-colorret-v25 --code-revision stage25-bmdraw-colorret-v25 \
  --output-dir "$OUT"
