#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-context-v18.csv
OUT=$BASE/runs/stage18/ATTR-V16-MNV3-256-BMDCONTEXT-BODY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 6 --batch-size 32 --workers 8 --learning-rate 3e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.35 --color-focal-gamma 0.0 \
  --class-weighting none --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.5 --small-sample-weight 2.0 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 1 --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-bmdcontext-body-v18 --code-revision stage18-bmdcontext-body-v18 \
  --output-dir "$OUT"
