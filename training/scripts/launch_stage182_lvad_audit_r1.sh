#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
auditor=/root/autodl-tmp/vcas/code/stage182-lvad-r1/audit_stage182_lvad_train.py
dataset_root='/root/autodl-tmp/vcas/sources/stage182-lvad-v2/extract/L-VAD (Low-Light Vehicle & Annotation Dataset)'
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE182-LVAD-AUDIT-R1

mkdir -p "${output_root}"
exec > "${output_root}/run.log" 2>&1

test -x "${python_bin}"
test -f "${auditor}"
test -d "${dataset_root}/train/images"
test -d "${dataset_root}/train/labels"

"${python_bin}" -m py_compile "${auditor}"
"${python_bin}" "${auditor}" \
  --root "${dataset_root}" \
  --report "${output_root}/stage182-lvad-train-audit.json" \
  --manifest "${output_root}/stage182-lvad-train-crops.csv" \
  --contact-sheet "${output_root}/stage182-lvad-numeric-class-contact-sheet.jpg"

sha256sum \
  "${output_root}/stage182-lvad-train-audit.json" \
  "${output_root}/stage182-lvad-train-crops.csv" \
  "${output_root}/stage182-lvad-numeric-class-contact-sheet.jpg" \
  > "${output_root}/SHA256SUMS"
