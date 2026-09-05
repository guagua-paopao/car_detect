#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/dataset-v1"
stage_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage53-color-v1"

for path in \
  "${dataset_root}/attributes/attribute_manifest_codex_labeled.csv" \
  "${dataset_root}/dataset-pilot-det-v1.json" \
  "${dataset_root}/openimages_attribution.csv"; do
  test -f "${path}" || { echo "fail_closed: missing ${path}" >&2; exit 2; }
done
test ! -e "${stage_root}"
mkdir -p "${stage_root}"

exec "${python_bin}" "${project_root}/training/scripts/build_openimages_foreground_color_proposals.py" \
  --manifest "${dataset_root}/attributes/attribute_manifest_codex_labeled.csv" \
  --dataset-root "${dataset_root}/attributes" \
  --source-card "${dataset_root}/dataset-pilot-det-v1.json" \
  --attribution "${dataset_root}/openimages_attribution.csv" \
  --output-manifest "${stage_root}/foreground-proposals.csv" \
  --output-report "${stage_root}/foreground-proposal-report.json"
