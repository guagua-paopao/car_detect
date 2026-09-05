#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
OUT="$ROOT/datasets/attribute-domain-v2/stage89-mio-balanced-expansion-v1"
LOG="$ROOT/runs/attributes/ATTR-STAGE89-MIO-BALANCED.log"
SESSION=VCAS-STAGE89-MIO-BALANCED

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already running: $SESSION"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing Stage89 output: $OUT" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "exec /root/miniconda3/bin/python '$ROOT/code/training/scripts/build_stage89_mio_balanced_expansion.py' \
    --archive '$ROOT/sources/mio-tcd/MIO-TCD-Classification.tar' \
    --expected-archive-sha256 0cf70c660f399aef05d069d9c993e2e3f89e887c823c781d3064fee3b2e6979c \
    --base-manifest '$ROOT/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv' \
    --expected-base-manifest-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
    --labels '$ROOT/code/config/vehicle_labels.v2.json' \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --output-images '$OUT/images' \
    --output-manifest '$OUT/attribute_manifest.stage89-mio-balanced.csv' \
    --output-report '$OUT/stage89-mio-balanced-report.json' \
    >'$LOG' 2>&1"

echo "started: $SESSION"
