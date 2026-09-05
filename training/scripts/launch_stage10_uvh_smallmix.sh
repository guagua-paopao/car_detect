#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh-small-mixture-v10.csv
OUT=$BASE/runs/stage10/ATTR-V8-MNV3-256-UVHSMALL-MIX
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 64 --workers 8 --learning-rate 8e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.45 --body-loss-weight 1.25 --focal-gamma 0.5 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 1.0 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --teacher-checkpoint $BASE/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.05 --distill-temperature 2.5 --freeze-backbone-epochs 1 \
  --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-uvh-small-mixture-v10 --code-revision stage10-uvh-smallmix-v10 \
  --output-dir "$OUT"
