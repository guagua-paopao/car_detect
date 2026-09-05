#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
source_root="/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging"
exclude_root="/root/autodl-tmp/vcas/datasets/dataset-v1"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
stage_root="${dataset_root}/openimages-stage54-hard-color-v1"

for path in "${source_root}/dataset-large-v1.json" "${exclude_root}/dataset-pilot-det-v1.json"; do
  test -f "${path}" || { echo "fail_closed: missing ${path}" >&2; exit 2; }
done
test ! -e "${stage_root}"

exec "${python_bin}" "${project_root}/training/scripts/build_openimages_large_hard_color_proposals.py" \
  --source-card "${source_root}/dataset-large-v1.json" \
  --source-root "${source_root}" \
  --exclude-card "${exclude_root}/dataset-pilot-det-v1.json" \
  --output-root "${stage_root}" \
  --output-manifest "${stage_root}/foreground-proposals.csv" \
  --output-report "${stage_root}/foreground-proposal-report.json"
