#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v23.csv
OUT=$BASE/runs/stage23/ATTR-V23-MNV3-256-BMDRAW-TYPE
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 6 --batch-size 64 --workers 8 --learning-rate 8e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.25 --color-focal-gamma 0.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.25 --small-sample-weight 0.75 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 1 --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-bmdraw-type-v23 --code-revision stage23-bmdraw-type-v23 \
  --output-dir "$OUT"
