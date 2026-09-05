#!/usr/bin/env bash
set -euo pipefail
MANIFEST=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.uvh26-hard-v3.csv
REPORT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.uvh26-hard-v3.report.json
OUT=/root/autodl-tmp/vcas/runs/stage5/ATTR-V3-MNV3-256-UVH26-BODY
mkdir -p "$(dirname "$OUT")"
while [ ! -s "$REPORT" ] || [ ! -s "$MANIFEST" ]; do sleep 15; done
exec /root/miniconda3/bin/python /root/autodl-tmp/vcas/code/training/scripts/train_attribute.py \
  --manifest "$MANIFEST" \
  --labels /root/autodl-tmp/vcas/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 16 --batch-size 64 --workers 8 --learning-rate 2e-4 --weight-decay 1e-4 \
  --color-loss-weight 0.7 --body-loss-weight 1.3 --focal-gamma 0.8 \
  --color-focal-gamma 0.3 --class-weighting inverse_sqrt --label-smoothing 0.05 \
  --type-threshold 0.75 --color-threshold 0.70 --device cuda \
  --augmentation-profile hard_scene --hard-sample-weight 2.0 \
  --init-checkpoint /root/autodl-tmp/vcas/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --teacher-checkpoint /root/autodl-tmp/vcas/runs/stage4/ATTR-V2-EFFV2S-256-HARD/best.pt \
  --distill-weight 0.15 --distill-temperature 2.5 \
  --run-kind formal --dataset-version attribute-domain-v2-uvh26-body-v3 \
  --code-revision stage5-uvh26-body-v3 --output-dir "$OUT"
