#!/usr/bin/env bash
set -euo pipefail

source_audit_session="vcas_cc0_audit_chain"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
source_root="${dataset_root}/kaggle-color-cc0-v1"
source_manifest="${source_root}/attribute_supplement.csv"
source_report="${source_root}/audit-report.json"
output_dir="${source_root}/contact-sheets"

while tmux has-session -t "${source_audit_session}" 2>/dev/null; do
  sleep 20
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
test ! -e "${output_dir}"

exec "${python_bin}" "${project_root}/training/scripts/build_attribute_audit_mosaics.py" \
  --manifest "${source_manifest}" \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --field color \
  --per-group 24 \
  --columns 6 \
  --seed 20260824
