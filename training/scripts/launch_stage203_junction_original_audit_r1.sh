#!/usr/bin/env bash
set -euo pipefail

session=VCAS-STAGE203-JUNCTION-AUDIT-R1
python_bin=/root/miniconda3/bin/python
source_root=/root/autodl-tmp/vcas/sources/stage203-junction-v1
output_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE203-JUNCTION-ORIGINAL-AUDIT-R1
script=/root/autodl-tmp/vcas/scripts/audit_stage203_junction_originals.py
log=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE203-JUNCTION-ORIGINAL-AUDIT-R1.log

[[ "$(readlink -m -- "${source_root}")" == "${source_root}" ]]
[[ "$(readlink -m -- "${output_root}")" == "${output_root}" ]]
[[ "${source_root}" == /root/autodl-tmp/vcas/sources/* ]]
[[ "${output_root}" == /root/autodl-tmp/vcas/runs/attributes/* ]]
[[ -x "${python_bin}" ]]
[[ -f "${script}" && ! -L "${script}" ]]
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "session already exists: ${session}" >&2
  exit 9
fi
mkdir -p -- "${source_root}" "$(dirname -- "${output_root}")"
tmux new-session -d -s "${session}" \
  "'${python_bin}' '${script}' --source-root '${source_root}' --output-root '${output_root}' --workers 8 >'${log}' 2>&1"
printf 'started session=%s log=%s\n' "${session}" "${log}"
