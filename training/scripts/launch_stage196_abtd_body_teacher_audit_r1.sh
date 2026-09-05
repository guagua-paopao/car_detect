#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage196-abtd-body-teacher-r1
stage195_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE195-ABTD-TRAIN-IMAGE-AUDIT-R2
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE196-ABTD-BODY-TEACHER-R2
labels=/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json
export PYTHONPATH=/root/autodl-tmp/vcas/code/training

test -f "${stage195_root}/stage195-abtd-train-image-audit.json"
test -f "${stage195_root}/stage195-abtd-train-crops.csv"
test ! -e "${output_root}"
mkdir -p "${output_root}"

"${python_bin}" "${code_root}/prepare_stage196_abtd_body_teacher_input.py" \
  --manifest "${stage195_root}/stage195-abtd-train-crops.csv" \
  --output "${output_root}/body-teacher-input.csv"

set +e
"${python_bin}" "${code_root}/audit_body_multiteacher_consensus.py" \
  --manifest "${output_root}/body-teacher-input.csv" --dataset-root / --labels "${labels}" \
  --checkpoint stage163_balanced=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1/ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1/best.pt \
  --checkpoint stage167_stretch=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/best.pt \
  --output-manifest "${output_root}/body-teacher-reviewed.csv" \
  --output-report "${output_root}/body-teacher-report.json" \
  --confidence 0.55 --minimum-total 300 --minimum-classes 2 \
  --batch-size 48 --workers 4 --device cuda \
  >"${output_root}/body-teacher.log" 2>&1
audit_rc=$?
set -e

if [[ $audit_rc -ne 0 && $audit_rc -ne 2 ]] || [[ ! -s "${output_root}/body-teacher-report.json" ]]; then
  echo "Stage196 teacher audit failed without complete evidence: rc=${audit_rc}" >&2
  exit 4
fi

sha256sum \
  "${output_root}/body-teacher-input.csv" \
  "${output_root}/body-teacher-reviewed.csv" \
  "${output_root}/body-teacher-report.json" \
  > "${output_root}/SHA256SUMS"

echo "STAGE196_TERMINAL audit_rc=${audit_rc}"
exit "${audit_rc}"
