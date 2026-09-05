#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh-small-mixture-v10.csv
OUT=$BASE/runs/stage16/ATTR-V14-MNV3-256-SMALLWEIGHT-BODY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 32 --workers 8 --learning-rate 4e-5 --weight-decay 1e-4 \
  --color-loss-weight 0.0 --body-loss-weight 1.0 --focal-gamma 0.5 --color-focal-gamma 0.0 \
  --class-weighting none --label-smoothing 0.02 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.75 --small-sample-weight 6.0 \
  --init-checkpoint $BASE/runs/stage4/ATTR-V2-MNV3-256-HARD/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-smallweighted-body-v16 --code-revision stage16-smallweighted-body-v16 \
  --output-dir "$OUT"
