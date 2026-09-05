#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage188-myvid-semantics-decontamination-r1
stage177_manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
stage187_manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE187-MYVID-V2-TRAIN-REAUDIT-R1/stage187-myvid-v2-train-crops.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1

test -x "${python_bin}"
test -f "${stage177_manifest}"
test -f "${stage187_manifest}"
test -f "${code_root}/stage188-myvid-v2-semantics.json"
test ! -e "${output_root}"
mkdir -p "${output_root}"
exec > "${output_root}/run.log" 2>&1

"${python_bin}" -m py_compile "${code_root}/audit_stage188_myvid_semantics_decontamination.py"
"${python_bin}" "${code_root}/audit_stage188_myvid_semantics_decontamination.py" \
  --stage187-manifest "${stage187_manifest}" \
  --existing-manifest "${stage177_manifest}" \
  --semantics "${code_root}/stage188-myvid-v2-semantics.json" \
  --output-manifest "${output_root}/stage188-myvid-v2-component-manifest.csv" \
  --report "${output_root}/stage188-myvid-v2-semantics-decontamination.json"

cp "${code_root}/stage188-myvid-v2-semantics.json" "${output_root}/"
sha256sum \
  "${output_root}/stage188-myvid-v2-component-manifest.csv" \
  "${output_root}/stage188-myvid-v2-semantics-decontamination.json" \
  "${output_root}/stage188-myvid-v2-semantics.json" \
  > "${output_root}/SHA256SUMS"
