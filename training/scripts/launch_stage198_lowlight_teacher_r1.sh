#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage198-lowlight-teacher-r1
manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE198-LOWLIGHT-TEACHER-R1

test -f "${manifest}"
test ! -e "${output_root}"
test "$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')" -ge 4000000000

"${python_bin}" -m py_compile "${code_root}/train_apply_stage198_lowlight_teacher.py"
set +e
"${python_bin}" "${code_root}/train_apply_stage198_lowlight_teacher.py" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 e6f66dcee1d9712816156168d0cbcfeff5e06a984e156bd9214293a8250d4153 \
  --allowed-root /root/autodl-tmp/vcas \
  --output-root "${output_root}" \
  --input-size 160 --epochs 5 --batch-size 128 --workers 8 \
  --validation-fraction 0.20 --minimum-precision 0.98 --minimum-validation-predictions 100 \
  --minimum-group-size 3 --minimum-group-positive-fraction 0.80 \
  --negative-ratio 4 --negative-floor-per-source 500 --seed 198 \
  >"${output_root}.log" 2>&1
audit_rc=$?
set -e

if [[ $audit_rc -ne 0 && $audit_rc -ne 2 ]] || [[ ! -s "${output_root}/stage198-lowlight-teacher-audit.json" ]]; then
  echo "Stage198 failed without complete evidence: rc=${audit_rc}" >&2
  exit 4
fi

sha256sum \
  "${output_root}/best.pt" \
  "${output_root}/stage198-unknown-lowlight-overlay.csv" \
  "${output_root}/stage198-lowlight-teacher-audit.json" \
  > "${output_root}/SHA256SUMS"

echo "STAGE198_TERMINAL audit_rc=${audit_rc}"
exit "${audit_rc}"
