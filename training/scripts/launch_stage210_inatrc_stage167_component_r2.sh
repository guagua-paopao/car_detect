#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT=${ROOT}/code/training/scripts/audit_stage210_inatrc_stage167_component.py
TEST=${ROOT}/code/training/tests/test_audit_stage210_inatrc_stage167_component.py
OUT=${ROOT}/runs/attributes/ATTR-STAGE210-INATRC-STAGE167-COMPONENT-R2

test ! -e "${OUT}"
"${PY}" -m unittest -v "${TEST}"
"${PY}" -m py_compile "${SCRIPT}"
"${PY}" "${SCRIPT}" \
  --stage167 "${ROOT}/datasets/attribute-domain-v2/stage167-complex-body-r1/attribute_manifest.stage167-complex-body.csv" \
  --expected-stage167-sha256 951937ebb28655aecdef2a0e655f67653175da7de030129ba7a073bedc515a32 \
  --existing-unlabeled "${ROOT}/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-unlabeled.csv" \
  --expected-unlabeled-sha256 f715881cfdc2a3d9005038822036588f612d887c6697af966abd5b7d0ebc5c04 \
  --stage91-all "${ROOT}/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/attribute_manifest.stage91-all.csv" \
  --expected-stage91-all-sha256 d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1 \
  --stage91-report "${ROOT}/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/stage91-inatrc-train-crops-report.json" \
  --expected-stage91-report-sha256 45585cc56e6d481145e59049113634d6fcf7179f694944f538baaeb0fdb2bfb0 \
  --output-dir "${OUT}" \
  > "${ROOT}/runs/attributes/ATTR-STAGE210-INATRC-STAGE167-COMPONENT-R2.log" 2>&1
