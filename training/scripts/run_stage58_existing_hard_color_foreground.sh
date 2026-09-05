#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
source_root="${dataset_root}/openimages-stage57-existing-hard-v1"
stage_root="${dataset_root}/openimages-stage58-existing-hard-color-v1"
large_root="/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging"

test -f "${source_root}/attribute-proposals.csv"
test ! -e "${stage_root}"
mkdir -p "${stage_root}"

exec "${python_bin}" "${project_root}/training/scripts/build_openimages_foreground_color_proposals.py" \
  --manifest "${source_root}/attribute-proposals.csv" \
  --dataset-root "${dataset_root}" \
  --source-card "/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage56-hard-box-plan-v1/hard-box-plan.csv" \
  --attribution "${large_root}/openimages_attribution.csv" \
  --output-manifest "${stage_root}/foreground-proposals.csv" \
  --output-report "${stage_root}/foreground-proposal-report.json"
