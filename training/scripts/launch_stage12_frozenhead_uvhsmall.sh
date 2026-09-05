#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh-small-mixture-v10.csv
OUT=$BASE/runs/stage12/ATTR-V10-MNV3-256-FROZENHEAD-UVHSMALL
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 6 --batch-size 64 --workers 8 --learning-rate 1e-4 --weight-decay 1e-4 \
  --color-loss-weight 0.30 --body-loss-weight 1.10 --focal-gamma 0.0 --color-focal-gamma 0.0 \
  --class-weighting none --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.5 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 6 --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-uvh-small-frozenhead-v12 --code-revision stage12-frozenhead-v12 \
  --output-dir "$OUT"
