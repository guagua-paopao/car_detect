#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage187-myvid-v2-train-reaudit-r1
source_root=/root/autodl-tmp/vcas/sources/stage185-myvid-v2
dataset_root="${source_root}/extract-train-only/MY-VID_v2.0_2025-12-09"
stage186_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE186-MYVID-V2-AUDIT-R1
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE187-MYVID-V2-TRAIN-REAUDIT-R1

test -x "${python_bin}"
test -f "${stage186_root}/stage186-myvid-v2-archive-audit.json"
test -d "${dataset_root}/train/images"
test -d "${dataset_root}/train/labels"
test ! -e "${output_root}"
mkdir -p "${output_root}"
exec > "${output_root}/run.log" 2>&1

"${python_bin}" -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); assert report["status"] == "pass"; assert report["scope"]["validation_images_opened"] == 0; assert report["scope"]["test_images_opened"] == 0; assert report["scope"]["frozen_video_used"] is False' \
  "${stage186_root}/stage186-myvid-v2-archive-audit.json"
"${python_bin}" -m py_compile "${code_root}/audit_stage186_myvid_v2_train.py"
"${python_bin}" "${code_root}/audit_stage186_myvid_v2_train.py" \
  --root "${dataset_root}" \
  --report "${output_root}/stage187-myvid-v2-train-reaudit.json" \
  --manifest "${output_root}/stage187-myvid-v2-train-crops.csv" \
  --contact-sheet "${output_root}/stage187-myvid-v2-numeric-class-contact-sheet.jpg"

sha256sum \
  "${output_root}/stage187-myvid-v2-train-reaudit.json" \
  "${output_root}/stage187-myvid-v2-train-crops.csv" \
  "${output_root}/stage187-myvid-v2-numeric-class-contact-sheet.jpg" \
  > "${output_root}/SHA256SUMS"
