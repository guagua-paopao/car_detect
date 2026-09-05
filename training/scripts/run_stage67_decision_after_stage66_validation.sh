#!/usr/bin/env bash
set -euo pipefail

upstream_session="vcas_stage66_licensed_adverse_validation"
validation_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-VALIDATION-V1.state.json"
validation_report="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-VALIDATION-V1/validation-screen-report.json"
decision_script="/root/autodl-tmp/vcas/code/training_stage66/scripts/decide_stage67_uvh26_followup.py"
decision_output="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-FOLLOWUP-DECISION-V1.json"
log_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-FOLLOWUP-DECISION-V1.log"
python_bin="/root/miniconda3/bin/python"

exec >>"${log_path}" 2>&1
while tmux has-session -t "${upstream_session}" 2>/dev/null; do
  sleep 30
done

test -f "${validation_state}"
test -f "${validation_report}"
test -f "${decision_script}"
test ! -e "${decision_output}"

exec "${python_bin}" "${decision_script}" \
  --validation-state "${validation_state}" \
  --validation-report "${validation_report}" \
  --output "${decision_output}"
