#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-type-v22.csv
OUT=$BASE/runs/stage22/ATTR-V22-MNV3-256-BMDTYPE-STRICT
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 10 --batch-size 32 --workers 8 --learning-rate 5e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.35 --color-focal-gamma 0.0 \
  --class-weighting none --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.75 --small-sample-weight 1.5 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-bmdtype-strict-v22 --code-revision stage22-bmdtype-strict-v22 \
  --output-dir "$OUT"
