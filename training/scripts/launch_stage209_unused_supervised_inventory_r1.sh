#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
RUN_ROOT=${ROOT}/runs/attributes/ATTR-STAGE209-UNUSED-SUPERVISED-INVENTORY-R1
SCRIPT=${ROOT}/code/training/scripts/audit_stage209_unused_supervised_inventory.py
PYTHON=/root/miniconda3/bin/python

mkdir -p "${RUN_ROOT}"
"${PYTHON}" "${SCRIPT}" \
  --dataset-root "${ROOT}/datasets/attribute-domain-v2" \
  --stage167 "${ROOT}/datasets/attribute-domain-v2/stage167-complex-body-r1/attribute_manifest.stage167-complex-body.csv" \
  --extra "${ROOT}/runs/attributes/ATTR-STAGE192-TESLA-TEACHER-QUOTA-R2/stage192-tesla-reviewed-manifest.csv" \
  --extra "${ROOT}/runs/attributes/ATTR-STAGE196-ABTD-BODY-TEACHER-R2/body-teacher-reviewed.csv" \
  --output-dir "${RUN_ROOT}" \
  > "${RUN_ROOT}/stage209.log" 2>&1
