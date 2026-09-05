#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
plan_root="${dataset_root}/openimages-stage56-hard-box-plan-v1"
output_root="${dataset_root}/openimages-stage57-existing-hard-v1"

test -f "${plan_root}/source-plan-report.json"
test -f "${plan_root}/hard-box-plan.csv"
test ! -e "${output_root}"

exec "${python_bin}" "${project_root}/training/scripts/build_openimages_existing_hard_crops.py" \
  --plan "${plan_root}/hard-box-plan.csv" \
  --output-root "${output_root}" \
  --dataset-root "${dataset_root}" \
  --output-manifest "${output_root}/attribute-proposals.csv" \
  --output-report "${output_root}/crop-build-report.json" \
  --margin 0.08 \
  --max-boxes-per-image 3 \
  --minimum-side-pixels 16 \
  --minimum-area-pixels 400
