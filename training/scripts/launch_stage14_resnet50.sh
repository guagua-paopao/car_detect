#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.hard-v2.csv
OUT=$BASE/runs/stage14/ATTR-V12-RESNET50-256-HARD
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture resnet50 --resize-mode stretch \
  --epochs 6 --batch-size 32 --workers 8 --learning-rate 1e-4 --weight-decay 1e-4 \
  --color-loss-weight 0.50 --body-loss-weight 1.25 --focal-gamma 0.5 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 1.0 \
  --teacher-checkpoint $BASE/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.05 --distill-temperature 2.5 --freeze-backbone-epochs 1 \
  --patience 2 --run-kind formal \
  --dataset-version attribute-domain-v2-resnet50-hard-v14 --code-revision stage14-resnet50-v14 \
  --output-dir "$OUT"
