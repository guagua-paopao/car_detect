#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
stage_root="${dataset_root}/openimages-stage57-existing-hard-v1"

test -f "${stage_root}/crop-build-report.json"
test -f "${stage_root}/attribute-proposals.csv"
test ! -e "${stage_root}/attribute-proposals.body-consensus.csv"
test ! -e "${stage_root}/body-teacher-consensus-report.json"

exec "${python_bin}" "${project_root}/training/scripts/pseudolabel_body_multiteacher_consensus.py" \
  --manifest "${stage_root}/attribute-proposals.csv" \
  --dataset-root "${dataset_root}" \
  --labels "${project_root}/config/vehicle_labels.v1.json" \
  --checkpoint "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --checkpoint "efficient=/root/autodl-tmp/vcas/runs/stage7/ATTR-V5-EFFV2S-256-UVH26-BODY/best.pt" \
  --checkpoint "resnet=/root/autodl-tmp/vcas/runs/stage14/ATTR-V12-RESNET50-256-HARD/best.pt" \
  --checkpoint "convnext=/root/autodl-tmp/vcas/runs/stage20/ATTR-V17-CONVNEXT-TINY-256-HARD/best.pt" \
  --output-manifest "${stage_root}/attribute-proposals.body-consensus.csv" \
  --output-report "${stage_root}/body-teacher-consensus-report.json" \
  --confidence 0.60 \
  --batch-size 96 \
  --workers 6 \
  --device cuda
