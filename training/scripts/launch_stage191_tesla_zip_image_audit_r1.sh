#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage191-tesla-zip-image-audit-r1
source_root=/root/autodl-tmp/vcas/sources/stage189-tesla-lighting-color
archive="${source_root}/Tesla-dataset.zip"
labels="${source_root}/tesla_dataset_labels.csv"
detector=/root/autodl-tmp/vcas/weights/yolo11x.pt
stage177=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv
stage188=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-component-manifest.csv
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE191-TESLA-ZIP-IMAGE-AUDIT-R1

while tmux has-session -t VCAS-DL-TESLA190 2>/dev/null; do
  sleep 20
done
test -f "${archive}.sha256"
sha256sum -c "${archive}.sha256"
test -f "${labels}"
test -f "${detector}"
test -f "${stage177}"
test -f "${stage188}"
test ! -e "${output_root}"
test "$(df --output=avail -B1 "${source_root}" | tail -n 1 | tr -d ' ')" -ge 1200000000

"${python_bin}" -m py_compile "${code_root}/audit_stage191_tesla_zip_images.py"
"${python_bin}" "${code_root}/audit_stage191_tesla_zip_images.py" \
  --archive "${archive}" \
  --labels "${labels}" \
  --detector "${detector}" \
  --cross-source-manifest "${stage177}" \
  --cross-source-manifest "${stage188}" \
  --output-root "${output_root}" \
  --batch-size 8 \
  --image-size 960 \
  --confidence 0.25 \
  --cross-distance 2 \
  --device 0

sha256sum \
  "${output_root}/stage191-tesla-lighting-color-manifest.csv" \
  "${output_root}/stage191-tesla-lighting-color-contact-sheet.jpg" \
  "${output_root}/stage191-tesla-zip-image-audit.json" \
  > "${output_root}/SHA256SUMS"
