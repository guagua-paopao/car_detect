#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT=${ROOT}/code/training/scripts/build_stage211_inatrc_toll_hardclass_manifests.py
TEST=${ROOT}/code/training/tests/test_build_stage211_inatrc_toll_hardclass_manifests.py
OUT=${ROOT}/datasets/attribute-domain-v2/stage211-inatrc-toll-hardclass-r1
STAGE210=${ROOT}/runs/attributes/ATTR-STAGE210-INATRC-STAGE167-COMPONENT-R2
LOG=${ROOT}/runs/attributes/ATTR-STAGE211-INATRC-TOLL-HARDCLASS-MANIFEST-R1.log

test ! -e "${OUT}"
"${PY}" -m unittest -v "${TEST}"
"${PY}" -m py_compile "${SCRIPT}"
"${PY}" "${SCRIPT}" \
  --base-manifest "${ROOT}/datasets/attribute-domain-v2/stage167-complex-body-r1/attribute_manifest.stage167-complex-body.csv" \
  --expected-base-sha256 951937ebb28655aecdef2a0e655f67653175da7de030129ba7a073bedc515a32 \
  --base-unlabeled "${ROOT}/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-unlabeled.csv" \
  --expected-base-unlabeled-sha256 f715881cfdc2a3d9005038822036588f612d887c6697af966abd5b7d0ebc5c04 \
  --stage210-supervised "${STAGE210}/attribute_manifest.stage210-inatrc-supervised.csv" \
  --expected-stage210-supervised-sha256 4855811eb5bdd6e5f9e2a1be21be36967abd2baf33c9ba2d917f0a729bef3565 \
  --stage210-unlabeled "${STAGE210}/attribute_manifest.stage210-inatrc-unlabeled.csv" \
  --expected-stage210-unlabeled-sha256 3f37e3a90559bc44b6dc82c5ca710206c24efaf9912440737f109891f61428f0 \
  --stage210-report "${STAGE210}/stage210-inatrc-stage167-component-audit.json" \
  --expected-stage210-report-sha256 121d48525fa3dec1ea1bf0ef9780103c0cb26a0365012b2dd95baff44912c274 \
  --output-dir "${OUT}" \
  > "${LOG}" 2>&1
