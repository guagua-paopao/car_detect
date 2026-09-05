#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.hard-v2.csv
OUT=$BASE/runs/stage11/ATTR-V9-MNV3-320-HARD
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 320 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 6 --batch-size 32 --workers 8 --learning-rate 6e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.50 --body-loss-weight 1.30 --focal-gamma 0.5 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 1.0 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --teacher-checkpoint $BASE/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.05 --distill-temperature 2.5 --freeze-backbone-epochs 1 \
  --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-320-hard-v11 --code-revision stage11-mnv3-320-v11 \
  --output-dir "$OUT"
