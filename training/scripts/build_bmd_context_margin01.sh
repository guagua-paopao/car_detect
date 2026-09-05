#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
exec /root/miniconda3/bin/python $BASE/code/training/scripts/build_bmd_context_crops.py \
  --manifest $BASE/datasets/attribute-domain-v2/attribute_manifest.hard-v2.csv \
  --dataset-root $BASE/datasets/dataset-domain-bmd45-v1 \
  --output-manifest $BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-context-v19.csv \
  --crop-root $BASE/datasets/attribute-domain-v2/bmd-context-crops-v19 \
  --margin 0.10 --max-hamming 14 --max-mse 45
