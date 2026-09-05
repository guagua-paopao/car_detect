#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
script=/root/autodl-tmp/vcas/scripts/inspect_stage208_stage206_unlabeled_overlap.py
stage206=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE206-UADETRAC-NIGHT-SUPPLEMENT-AUDIT-R2/stage206-uadetrac-night-supplement.accepted.csv
existing=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-unlabeled.csv
output=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE208-STAGE206-UNLABELED-OVERLAP-R1/stage208-stage206-existing-unlabeled-overlap.json

[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
[[ -f "${stage206}" && ! -L "${stage206}" ]]
[[ -f "${existing}" && ! -L "${existing}" ]]
[[ ! -e "${output}" ]]
"${python_bin}" -m py_compile "${script}"
"${python_bin}" "${script}" --stage206 "${stage206}" --existing-unlabeled "${existing}" --output "${output}"
