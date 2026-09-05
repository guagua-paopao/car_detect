#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/vcas
code_root="${base}/code/stage218_all_color_train_metadata_r1"
auditor="${code_root}/audit_stage218_all_color_train_metadata.py"
test_file="${code_root}/test_audit_stage218_all_color_train_metadata.py"
dataset_root="${base}/datasets/attribute-domain-v2"
output_root="${base}/runs/attributes/ATTR-STAGE218-ALL-COLOR-TRAIN-METADATA-R1"
outer_log="${base}/runs/attributes/ATTR-STAGE218-ALL-COLOR-TRAIN-METADATA-R1.log"

test ! -e "${output_root}"
test -d "${dataset_root}"
test "$(sha256sum "${auditor}" | awk '{print $1}')" = "a2a6f7137ae13aae1aecf692e8249fcd17330d9f96ec32d20a25bce3db943a2a"
test "$(sha256sum "${test_file}" | awk '{print $1}')" = "462d279e5f571a8393c3b5956b440ee0b51c66fc85c7bdf1e7010fe4c36fb500"

available_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
test -n "${available_bytes}"
test "${available_bytes}" -ge 12884901888

exec > >(tee -a "${outer_log}") 2>&1
cd "${code_root}"
/root/miniconda3/bin/python -m py_compile "${auditor}"
/root/miniconda3/bin/python -m unittest -v test_audit_stage218_all_color_train_metadata.py

/root/miniconda3/bin/python "${auditor}" \
  --root "${dataset_root}" \
  --output-root "${output_root}" \
  --max-manifests 500

sha256sum \
  "${output_root}/stage218-authoritative-color-unique-metadata.csv" \
  "${output_root}/stage218-all-color-train-metadata-audit.json" \
  "${outer_log}" \
  > "${output_root}/SHA256SUMS"

echo "STAGE218_COMPLETE"
