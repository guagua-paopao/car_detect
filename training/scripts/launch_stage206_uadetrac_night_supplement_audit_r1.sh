#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
script=/root/autodl-tmp/vcas/scripts/audit_stage206_uadetrac_night_supplement.py
stage70_root=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2
supplement_manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE205-UADETRAC-STAGE177-GAP-R2/stage205-uadetrac-night-gap-track-cap5.csv
stage177_manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE206-UADETRAC-NIGHT-SUPPLEMENT-AUDIT-R1

[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
[[ -d "${stage70_root}" && ! -L "${stage70_root}" ]]
[[ -f "${supplement_manifest}" && ! -L "${supplement_manifest}" ]]
[[ -f "${stage177_manifest}" && ! -L "${stage177_manifest}" ]]
[[ ! -e "${output_root}" ]]
"${python_bin}" -m py_compile "${script}"
"${python_bin}" "${script}" \
  --stage70-root "${stage70_root}" \
  --supplement-manifest "${supplement_manifest}" \
  --stage177-manifest "${stage177_manifest}" \
  --output-root "${output_root}"
