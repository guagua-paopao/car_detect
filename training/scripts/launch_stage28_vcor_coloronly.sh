#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v28-coloronly.csv
OUT=$BASE/runs/stage28/ATTR-V28-MNV3-224-VCORCOLOR-ONLY
mkdir -p "$(dirname "$OUT")"
exec /root/miniconda3/bin/python $BASE/code/training/scripts/train_attribute.py \
  --manifest "$MAN" --labels $BASE/code/config/vehicle_labels.v1.json \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 10 --batch-size 64 --workers 8 --learning-rate 2e-5 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 0.0 --focal-gamma 0.0 --color-focal-gamma 0.15 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --type-threshold 0.80 --color-threshold 0.80 \
  --device cuda --augmentation-profile hard_scene --hard-sample-weight 0.0 --small-sample-weight 0.0 --color-sample-weight 2.0 \
  --init-checkpoint $BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt \
  --freeze-backbone-epochs 1 --patience 3 --run-kind formal \
  --dataset-version attribute-domain-v2-mnv3-vcor-coloronly-v28 --code-revision stage28-vcor-coloronly-v28 \
  --output-dir "$OUT"
