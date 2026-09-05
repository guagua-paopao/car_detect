#!/usr/bin/env bash
set -euo pipefail

stage211_session=VCAS-STAGE211-INATRC-TOLL-HARDCLASS-R1
stage212_session=VCAS-STAGE212-STAGE211-BODY-VALIDATION-R1
stage214_launcher=/root/autodl-tmp/vcas/scripts/launch_stage214_current_train_scene_reaudit_r1.sh
log=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE214-CURRENT-TRAIN-SCENE-REAUDIT-R1.log

exec > >(tee -a "${log}") 2>&1

[[ -x "${stage214_launcher}" && ! -L "${stage214_launcher}" ]]

while tmux has-session -t "${stage211_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${stage212_session}" 2>/dev/null; do
  sleep 30
done

exec "${stage214_launcher}"
