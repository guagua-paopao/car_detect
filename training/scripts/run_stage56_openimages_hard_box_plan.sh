#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
large_root="/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging"
pilot_root="/root/autodl-tmp/vcas/datasets/dataset-v1"
cache_root="${large_root}/_openimages_cache/open-images-v7/train"
output_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage56-hard-box-plan-v1"

test ! -e "${output_root}"
mkdir -p "${output_root}"

exec "${python_bin}" "${project_root}/training/scripts/build_openimages_hard_box_plan.py" \
  --detections "${cache_root}/labels/detections.csv" \
  --image-metadata "${cache_root}/metadata/image_ids.csv" \
  --existing-card "${large_root}/dataset-large-v1.json" \
  --existing-card "${pilot_root}/dataset-pilot-det-v1.json" \
  --output-plan "${output_root}/hard-box-plan.csv" \
  --output-report "${output_root}/source-plan-report.json" \
  --small-area-ratio 0.01
