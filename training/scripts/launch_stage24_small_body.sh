#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.small-body-v24.csv
OUT=$BASE/runs/stage24/ATTR-V24-MNV3-320-SMALLBODY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 320 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 32 --workers 8 --learning-rate 5e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.35 --color-focal-gamma 0.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.5 --small-sample-weight 0.0 \
  --init-checkpoint $BASE/runs/stage23/ATTR-V23-MNV3-256-BMDRAW-TYPE/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-320-smallbody-v24 --code-revision stage24-smallbody-v24 \
  --output-dir "$OUT"
