#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
script=/root/autodl-tmp/vcas/scripts/audit_stage204_uadetrac_train_metadata.py
source_root=/root/autodl-tmp/vcas/sources/ua-detrac-hf-v1
stage177=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE204-UADETRAC-TRAIN-METADATA-AUDIT-R2

[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
[[ -d "${source_root}" && ! -L "${source_root}" ]]
[[ -f "${stage177}" && ! -L "${stage177}" ]]
[[ ! -e "${output_root}" ]]
"${python_bin}" -m py_compile "${script}"
"${python_bin}" "${script}" \
  --source-root "${source_root}" \
  --stage177-manifest "${stage177}" \
  --output-root "${output_root}"
