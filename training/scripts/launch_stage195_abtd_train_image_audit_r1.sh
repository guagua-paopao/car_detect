#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage195-abtd-train-image-audit-r1
source_root=/root/autodl-tmp/vcas/sources/stage193-abtd-v3
images="${source_root}/images.zip"
labels="${source_root}/labels.zip"
metadata="${source_root}/metadata.csv"
stage177=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
stage188=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-component-manifest.csv
stage191=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE191-TESLA-ZIP-IMAGE-AUDIT-R1/stage191-tesla-lighting-color-manifest.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE195-ABTD-TRAIN-IMAGE-AUDIT-R2

test -f "${images}.sha256"
(cd "${source_root}" && sha256sum -c images.zip.sha256)
test -f "${labels}"
test -f "${metadata}"
test -f "${stage177}"
test -f "${stage188}"
test -f "${stage191}"
test ! -e "${output_root}"
test "$(df --output=avail -B1 "${source_root}" | tail -n 1 | tr -d ' ')" -ge 3000000000

"${python_bin}" -m py_compile "${code_root}/audit_stage195_abtd_train_images.py"
"${python_bin}" "${code_root}/audit_stage195_abtd_train_images.py" \
  --images "${images}" \
  --labels "${labels}" \
  --metadata "${metadata}" \
  --cross-source-manifest "${stage177}" \
  --cross-source-manifest "${stage188}" \
  --cross-source-manifest "${stage191}" \
  --output-root "${output_root}" \
  --cross-distance 2

sha256sum \
  "${output_root}/stage195-abtd-train-crops.csv" \
  "${output_root}/stage195-abtd-contact-sheet.jpg" \
  "${output_root}/stage195-abtd-train-image-audit.json" \
  > "${output_root}/SHA256SUMS"
