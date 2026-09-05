#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/miniconda3/bin/python
code_root=/root/autodl-tmp/vcas/code/stage186-myvid-v2-audit-r1
source_root=/root/autodl-tmp/vcas/sources/stage185-myvid-v2
archive="${source_root}/MY-VID_v2.0_2025-12-09.zip"
extract_root="${source_root}/extract-train-only"
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE186-MYVID-V2-AUDIT-R1

mkdir -p "${output_root}"
exec > "${output_root}/run.log" 2>&1

while tmux has-session -t VCAS-DL-MYVID185 2>/dev/null; do
  sleep 20
done

test -x "${python_bin}"
test -f "${archive}"
test -f "${archive}.sha256"
test -f "${code_root}/audit_stage186_myvid_v2_archive.py"
test -f "${code_root}/audit_stage186_myvid_v2_train.py"
sha256sum -c "${archive}.sha256"

"${python_bin}" -m py_compile \
  "${code_root}/audit_stage186_myvid_v2_archive.py" \
  "${code_root}/audit_stage186_myvid_v2_train.py"
"${python_bin}" "${code_root}/audit_stage186_myvid_v2_archive.py" \
  --archive "${archive}" \
  --report "${output_root}/stage186-myvid-v2-archive-audit.json"

test ! -e "${extract_root}"
available_bytes="$(df --output=avail -B1 "${source_root}" | tail -n 1 | tr -d ' ')"
test -n "${available_bytes}"
test "${available_bytes}" -ge 8000000000
mkdir -p "${extract_root}"
unzip -q "${archive}" '*/train/*' '*/data.yaml' -d "${extract_root}"

mapfile -t dataset_roots < <(find "${extract_root}" -type f -name data.yaml -printf '%h\n')
test "${#dataset_roots[@]}" -eq 1
dataset_root="$(readlink -f "${dataset_roots[0]}")"
case "${dataset_root}" in
  "${extract_root}"/*) ;;
  *) echo "dataset root escaped extraction directory"; exit 3 ;;
esac
test -d "${dataset_root}/train/images"
test -d "${dataset_root}/train/labels"

"${python_bin}" "${code_root}/audit_stage186_myvid_v2_train.py" \
  --root "${dataset_root}" \
  --report "${output_root}/stage186-myvid-v2-train-audit.json" \
  --manifest "${output_root}/stage186-myvid-v2-train-crops.csv" \
  --contact-sheet "${output_root}/stage186-myvid-v2-numeric-class-contact-sheet.jpg"

sha256sum \
  "${output_root}/stage186-myvid-v2-archive-audit.json" \
  "${output_root}/stage186-myvid-v2-train-audit.json" \
  "${output_root}/stage186-myvid-v2-train-crops.csv" \
  "${output_root}/stage186-myvid-v2-numeric-class-contact-sheet.jpg" \
  > "${output_root}/SHA256SUMS"
