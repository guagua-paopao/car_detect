#!/usr/bin/env bash
set -euo pipefail

source_audit_session="vcas_cc0_audit_chain"
type_validation_session="vcas_stage50_validation"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
source_root="${dataset_root}/kaggle-color-cc0-v1"
source_manifest="${source_root}/attribute_supplement.csv"
source_report="${source_root}/audit-report.json"
output_manifest="${source_root}/attribute_supplement.teacher-approved-v1.csv"
output_report="${source_root}/teacher-audit-report.json"

while tmux has-session -t "${source_audit_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${type_validation_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${source_report}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
if not report_path.is_file():
    raise SystemExit("fail_closed: CC0 source audit report is missing")
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "pass":
    raise SystemExit(f"fail_closed: CC0 source audit status is {report.get('status')}")
PY

test -f "${source_manifest}"
test ! -e "${output_manifest}"
test ! -e "${output_report}"

exec "${python_bin}" "${project_root}/training/scripts/audit_color_multiteacher_consensus.py" \
  --manifest "${source_manifest}" \
  --dataset-root "${dataset_root}" \
  --labels "${project_root}/config/vehicle_labels.v1.json" \
  --checkpoint "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --checkpoint "stage44=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-BMD45-RAW-V34-256-R1/best.pt" \
  --checkpoint "stage47=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE47-HARDTYPE-UVH-256-R1/best.pt" \
  --output-manifest "${output_manifest}" \
  --output-report "${output_report}" \
  --confidence 0.50 \
  --minimum-total 100 \
  --minimum-classes 4 \
  --batch-size 128 \
  --workers 4 \
  --device cuda
