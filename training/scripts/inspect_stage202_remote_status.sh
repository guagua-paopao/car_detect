#!/usr/bin/env bash
set -euo pipefail

date -Is
df -B1 --output=size,used,avail,pcent,target /root/autodl-tmp | tail -n 1
printf '%s\n' 'TMUX_SESSIONS'
tmux ls 2>/dev/null || true
printf '%s\n' 'RELEVANT_PROCESSES'
ps -eo pid,etimes,cmd --sort=-etimes | grep -E 'ATTR-STAGE|stage19|stage20|train_apply' | grep -v grep || true
printf '%s\n' 'LATEST_STAGE_OUTPUTS'
find /root/autodl-tmp/vcas/runs/attributes -mindepth 1 -maxdepth 1 -type d \
  \( -name 'ATTR-STAGE19*' -o -name 'ATTR-STAGE20*' \) \
  -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS%Tz %f\n' | sort -nr | head -n 30
printf '%s\n' 'STAGE203_PROGRESS'
stage203_images=/root/autodl-tmp/vcas/sources/stage203-junction-v1/original_images
stage203_log=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE203-JUNCTION-ORIGINAL-AUDIT-R1.log
if [[ -d "${stage203_images}" ]]; then
  printf 'downloaded_files=%s downloaded_bytes=%s\n' \
    "$(find "${stage203_images}" -maxdepth 1 -type f -printf . | wc -c)" \
    "$(find "${stage203_images}" -maxdepth 1 -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum + 0}')"
fi
tail -n 5 "${stage203_log}" 2>/dev/null || true
