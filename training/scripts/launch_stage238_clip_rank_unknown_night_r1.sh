#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE238-CLIP-RANK-R1"
TRAINING_CODE="/root/autodl-tmp/vcas/code/training"
PYTHON="/root/autodl-tmp/vcas/venvs/stage237-clip/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE238-CLIP-RANK-UNKNOWN-NIGHT-R1"
INPUT_MANIFEST="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE179-FULL-COLOR-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv"
STAGE237_REPORT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE237-CLIP-UNKNOWN-NIGHT-R1/stage237-clip-unknown-night-audit.json"
WEIGHT="/root/autodl-tmp/vcas/weights/open_clip/ViT-B-32.pt"
LOG="/root/autodl-tmp/vcas/logs/stage238-clip-rank-unknown-night-r1.log"

run_worker() {
  "${PYTHON}" "${TRAINING_CODE}/scripts/audit_stage238_clip_rank_unknown_night.py" \
    --active-manifest "${INPUT_MANIFEST}" \
    --stage237-report "${STAGE237_REPORT}" \
    --model ViT-B-32-quickgelu --weight "${WEIGHT}" \
    --cache-dir /root/autodl-tmp/vcas/weights/open_clip \
    --output-ranking "${OUTPUT_ROOT}/stage238-clip-ranked-unknown-night.csv" \
    --output-report "${OUTPUT_ROOT}/stage238-clip-rank-unknown-night.json" \
    --output-contact-sheets "${OUTPUT_ROOT}/contact-sheets" \
    --per-source 100 --per-source-color 25 \
    --batch-size 256 --workers 8 --device cuda
  sha256sum \
    "${OUTPUT_ROOT}/stage238-clip-ranked-unknown-night.csv" \
    "${OUTPUT_ROOT}/stage238-clip-rank-unknown-night.json" \
    > "${OUTPUT_ROOT}/SHA256SUMS"
  find "${OUTPUT_ROOT}/contact-sheets" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 -r sha256sum >> "${OUTPUT_ROOT}/SHA256SUMS"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi
for required in "${PYTHON}" "${TRAINING_CODE}/scripts/audit_stage238_clip_rank_unknown_night.py" \
  "${TRAINING_CODE}/scripts/audit_stage237_clip_unknown_night.py" "${INPUT_MANIFEST}" \
  "${STAGE237_REPORT}" "${WEIGHT}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash '$0' --worker >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
