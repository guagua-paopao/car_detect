#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
RUN=$BASE/runs/stage5/ATTR-V3-MNV3-256-UVH26-COLORSAFE
MAN=$BASE/datasets/attribute-domain-v2/attribute_manifest.uvh26-color-safe-v4.csv
LABELS=$BASE/code/config/vehicle_labels.v1.json
/root/miniconda3/bin/python $BASE/code/training/scripts/calibrate_attribute_temperature.py \
  --checkpoint $RUN/best.pt --manifest $MAN --labels $LABELS \
  --output $RUN/calibration-v4-colorsafe.json --body-threshold 0.75 --color-threshold 0.70 \
  --target-precision 0.93 --device cuda
/root/miniconda3/bin/python $BASE/code/training/scripts/evaluate_attribute_baseline.py \
  --manifest $MAN --checkpoint $RUN/best.pt --labels $LABELS \
  --output $RUN/test-metrics-v4-colorsafe.json --split test --batch-size 64 --workers 4 \
  --device cuda --type-threshold 0.75 --color-threshold 0.70
echo EVALUATION_COMPLETE
