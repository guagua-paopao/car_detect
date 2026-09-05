#!/usr/bin/env bash
set -euo pipefail

teacher_session="vcas_stage54_oi_teacher_audit"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
stage_root="${dataset_root}/openimages-stage54-hard-color-v1"

while tmux has-session -t "=${teacher_session}" 2>/dev/null; do sleep 30; done

test -f "${stage_root}/teacher-audit-report.json"
exec "${python_bin}" "${project_root}/training/scripts/audit_stage53_openimages_color_domain.py" \
  --manifest "${stage_root}/teacher-approved.csv" \
  --teacher-report "${stage_root}/teacher-audit-report.json" \
  --dataset-root "${dataset_root}" \
  --source-card "/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging/dataset-large-v1.json" \
  --output-manifest "${stage_root}/domain-filtered-supplement.csv" \
  --output-report "${stage_root}/domain-audit-report.json" \
  --dhash-distance 4
