#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage207-uadetrac-night-teacher-r1
script=${code_root}/pseudolabel_stage207_uadetrac_night_multiteacher.py
manifest=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE206-UADETRAC-NIGHT-SUPPLEMENT-AUDIT-R2/stage206-uadetrac-night-supplement.accepted.csv
manifest_sha=241547c997c48a361c148c9af90e1df8f2ac3322a9959b9e5ea31c29611f799c
images_root=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2
labels=/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json
labels_sha=22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE207-UADETRAC-NIGHT-MULTITEACHER-R1
log=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE207-UADETRAC-NIGHT-MULTITEACHER-R1.log

stage161=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE161-BODY-FOCUS-DISTILL-R1/ATTR-STAGE161-BODY-CONVNEXT-256-FOCUS-R1/best.pt
stage163=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1/ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1/best.pt
stage167=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/best.pt
production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
effv2=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-B-EFFV2S-256/best.pt

[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
[[ -f "${manifest}" && ! -L "${manifest}" ]]
[[ -d "${images_root}" && ! -L "${images_root}" ]]
[[ -f "${labels}" && ! -L "${labels}" ]]
[[ ! -e "${output_root}" ]]
[[ ! -e "${log}" ]]
"${python_bin}" -m py_compile "${script}"
"${python_bin}" "${script}" \
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha}" \
  --images-root "${images_root}" \
  --labels "${labels}" --expected-labels-sha256 "${labels_sha}" \
  --checkpoint "stage161=${stage161}" --expected-checkpoint-sha256 "stage161=2b4cde3017d5a673dd22a5988cec3326c833106bbf6eefc21a1bf0cdeab48b2a" \
  --checkpoint "stage163=${stage163}" --expected-checkpoint-sha256 "stage163=5e8b95922cb0a2fac2f3cd289be7b00f30018cf2ccc8d0c43376308346720d5b" \
  --checkpoint "stage167=${stage167}" --expected-checkpoint-sha256 "stage167=c6dea9d50a01fd6395e403606083e955d8880f9f9e77048d0d5f28c487b11f0b" \
  --checkpoint "production=${production}" --expected-checkpoint-sha256 "production=6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383" \
  --checkpoint "effv2=${effv2}" --expected-checkpoint-sha256 "effv2=ad08a34c085ab73bbd2b7ef088af03b0651c2e6fbfd474b9c4f6591ef8098946" \
  --output-root "${output_root}" \
  --confidence 0.70 --minimum-track-frames 2 --minimum-track-strong-fraction 0.50 \
  --minimum-authorized-rows 100 --minimum-authorized-classes 3 \
  --batch-size 96 --workers 6 --device cuda >"${log}" 2>&1
