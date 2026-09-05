#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
script=/root/autodl-tmp/vcas/scripts/build_stage205_uadetrac_stage177_gap.py
stage70_root=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2
stage177=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE205-UADETRAC-STAGE177-GAP-R2

[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
[[ -d "${stage70_root}" && ! -L "${stage70_root}" ]]
[[ -f "${stage177}" && ! -L "${stage177}" ]]
[[ ! -e "${output_root}" ]]
"${python_bin}" -m py_compile "${script}"
"${python_bin}" "${script}" \
  --stage70-root "${stage70_root}" \
  --stage177-manifest "${stage177}" \
  --output-root "${output_root}" \
  --combined-track-cap 5
