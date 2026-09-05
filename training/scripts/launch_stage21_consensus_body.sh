#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-consensus-v21.csv
OUT=$BASE/runs/stage21/ATTR-V18-MNV3-256-BMDCONSENSUS-BODY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 32 --workers 8 --learning-rate 3e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.25 --color-focal-gamma 0.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.5 --small-sample-weight 2.5 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-bmdconsensus-body-v21 --code-revision stage21-bmdconsensus-body-v21 \
  --output-dir "$OUT"
