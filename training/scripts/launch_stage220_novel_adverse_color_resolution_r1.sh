#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/vcas
code_root="${base}/code/stage220_novel_adverse_color_r1"
resolver="${code_root}/resolve_stage220_novel_adverse_color_candidates.py"
test_file="${code_root}/test_resolve_stage220_novel_adverse_color_candidates.py"
candidates="${code_root}/stage219-novel-adverse-color-candidates.csv"
validation="${code_root}/color.validation-only.csv"
stage218_report="${base}/runs/attributes/ATTR-STAGE218-ALL-COLOR-TRAIN-METADATA-R1/stage218-all-color-train-metadata-audit.json"
output_root="${base}/runs/attributes/ATTR-STAGE220-NOVEL-ADVERSE-COLOR-RESOLUTION-R1"
outer_log="${base}/runs/attributes/ATTR-STAGE220-NOVEL-ADVERSE-COLOR-RESOLUTION-R1.log"

test ! -e "${output_root}"
test "$(sha256sum "${resolver}" | awk '{print $1}')" = "2941f96180c94cd15497240e67b99fd2373e5c5eef9efb5fdb5396cf3d0fc8b1"
test "$(sha256sum "${test_file}" | awk '{print $1}')" = "d673bc7d7700d8de25aec0018940a256a24cef6a48d1c10c2f3574aaa27290d6"
test "$(sha256sum "${candidates}" | awk '{print $1}')" = "accb79db9b3340929c4970cf1aa0cde6c287cc942416215f1d1fba05c24dc2d3"
test "$(sha256sum "${validation}" | awk '{print $1}')" = "34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023"
test "$(sha256sum "${stage218_report}" | awk '{print $1}')" = "c75da4a38d98f61f8a97a46ec202d5fa3a6257326d53f53004cf77a0c2dd78cd"

available_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
test -n "${available_bytes}"
test "${available_bytes}" -ge 12884901888

exec > >(tee -a "${outer_log}") 2>&1
cd "${code_root}"
/root/miniconda3/bin/python -m py_compile "${resolver}"
/root/miniconda3/bin/python -m unittest -v test_resolve_stage220_novel_adverse_color_candidates.py

/root/miniconda3/bin/python "${resolver}" \
  --candidates "${candidates}" \
  --expected-candidates-sha256 accb79db9b3340929c4970cf1aa0cde6c287cc942416215f1d1fba05c24dc2d3 \
  --stage218-report "${stage218_report}" \
  --expected-stage218-report-sha256 c75da4a38d98f61f8a97a46ec202d5fa3a6257326d53f53004cf77a0c2dd78cd \
  --validation-manifest "${validation}" \
  --expected-validation-sha256 34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023 \
  --allowed-root "${base}" \
  --output-root "${output_root}" \
  --near-duplicate-hamming 4

sha256sum \
  "${output_root}/attribute_manifest.stage220-novel-adverse-color-audited.csv" \
  "${output_root}/stage220-quarantined-candidates.csv" \
  "${output_root}/stage220-novel-adverse-color-resolution.json" \
  "${outer_log}" \
  > "${output_root}/SHA256SUMS"

echo "STAGE220_COMPLETE"
