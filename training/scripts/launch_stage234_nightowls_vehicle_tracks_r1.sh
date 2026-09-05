#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE234-NIGHTOWLS-TRACKS"
INPUT_ROOT="/root/autodl-tmp/vcas/sources/nightowls-stage233-pilot"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE234-NIGHTOWLS-VEHICLE-TRACKS-R1"
SCRIPT="/root/autodl-tmp/vcas/scripts/build_stage234_nightowls_vehicle_tracks.py"
DETECTOR="/root/autodl-tmp/vcas/weights/yolo11x.pt"
STAGE177="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv"
STAGE188="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-component-manifest.csv"
LOG="/root/autodl-tmp/vcas/logs/stage234-nightowls-vehicle-tracks-r1.log"
PYTHON="/root/miniconda3/bin/python3"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi
for required in \
  "${INPUT_ROOT}/stage233-nightowls-train-pilot-report.json" \
  "${INPUT_ROOT}/stage233-nightowls-train-pilot-manifest.csv" \
  "${SCRIPT}" "${DETECTOR}" "${STAGE177}" "${STAGE188}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
"${PYTHON}" - "${INPUT_ROOT}/stage233-nightowls-train-pilot-report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "complete_train_only_research_only"
assert report["selected_frames"] == 2400
assert report["recording_groups"] == 24
assert report["block_groups"] == 120
assert report["decode_errors"] == 0
assert report["exact_sha256_duplicates"] == 0
assert report["policy"]["validation_or_test_pixels_opened"] == 0
assert report["policy"]["frozen_video_used"] is False
PY

mkdir -p "$(dirname "${LOG}")"
INNER="set -euo pipefail; \
  '${PYTHON}' '${SCRIPT}' \
    --input-manifest '${INPUT_ROOT}/stage233-nightowls-train-pilot-manifest.csv' \
    --detector '${DETECTOR}' \
    --cross-source-manifest '${STAGE177}' \
    --cross-source-manifest '${STAGE188}' \
    --output-root '${OUTPUT_ROOT}' \
    --batch-size 8 --image-size 960 --device 0 --near-distance 2; \
  echo STAGE234_COMPLETE"
tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
