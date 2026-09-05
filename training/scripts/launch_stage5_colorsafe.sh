#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh26-color-safe-v4.csv
OUT=$BASE/runs/stage5/ATTR-V3-MNV3-256-UVH26-COLORSAFE
mkdir -p "$(dirname "$OUT")"
while [ ! -s "$MAN" ]; do sleep 10; done
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 12 --batch-size 64 --workers 8 --learning-rate 1.5e-4 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 1.0 --focal-gamma 0.5 --color-focal-gamma 0.2 \
  --class-weighting inverse_sqrt --label-smoothing 0.04 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.75 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --teacher-checkpoint $BASE/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.10 --distill-temperature 2.5 --run-kind formal \
  --dataset-version attribute-domain-v2-uvh26-colorsafe-v4 --code-revision stage5-colorsafe-v4 \
  --output-dir "$OUT"
