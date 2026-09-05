#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v27-vcorcolor.csv
OUT=$BASE/runs/stage27/ATTR-V27-MNV3-224-VCORCOLOR-ONLY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 64 --workers 8 --learning-rate 2e-5 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 0.0 --focal-gamma 0.0 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.80 --color-threshold 0.80 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.0 --small-sample-weight 0.0 --color-sample-weight 3.0 \
  --init-checkpoint $BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-vcorcolor-v27 --code-revision stage27-vcor-color-only-v27 \
  --output-dir "$OUT"
