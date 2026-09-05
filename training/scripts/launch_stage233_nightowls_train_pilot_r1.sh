#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-DL-NIGHTOWLS233"
SOURCE_ROOT="/root/autodl-tmp/vcas/sources/nightowls-stage232"
OUTPUT_ROOT="/root/autodl-tmp/vcas/sources/nightowls-stage233-pilot"
LOG="/root/autodl-tmp/vcas/logs/stage233-nightowls-train-pilot-r1.log"
SCRIPT="/root/download_stage233_nightowls_train_pilot.py"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}"
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "/root/miniconda3/bin/python3 '${SCRIPT}' \
    --url 'https://thor.robots.ox.ac.uk/nightowls/python/nightowls_training.zip' \
    --annotations '${SOURCE_ROOT}/nightowls_training.json' \
    --license-html '${SOURCE_ROOT}/official-download.html' \
    --output-root '${OUTPUT_ROOT}' \
    --blocks 5 \
    --block-size 20 \
    --max-compressed-gib 4.0 \
    --workers 4 \
    >>'${LOG}' 2>&1"
echo "started ${SESSION}"
echo "log ${LOG}"
