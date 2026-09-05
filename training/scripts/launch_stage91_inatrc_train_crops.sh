#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
OUT="$ROOT/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1"
LOG="$ROOT/runs/attributes/ATTR-STAGE91-INATRC-CROPS.log"
SESSION=VCAS-STAGE91-INATRC-CROPS

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already running: $SESSION"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing Stage91 output: $OUT" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "exec /root/miniconda3/bin/python '$ROOT/code/training/scripts/build_stage91_inatrc_train_crops.py' \
    --archive '$ROOT/sources/inatrc-stage91/inatrc-v1.zip' \
    --expected-archive-sha256 98ca8df90167a525eb60e123f8b31c33d16486b13d9f13a812dc3af9eb39b860 \
    --audit-report '$ROOT/sources/inatrc-stage91/stage91-inatrc-archive-audit-v1.json' \
    --expected-audit-report-sha256 ceb79fb94353fa44880c227fca4a40b2dbc4d41299b4fc70b7f9f9fbc7cd13fa \
    --base-manifest '$ROOT/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv' \
    --expected-base-manifest-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
    --labels '$ROOT/code/config/vehicle_labels.v2.json' \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --output-images '$OUT/images' \
    --output-all-manifest '$OUT/attribute_manifest.stage91-all.csv' \
    --output-supervised-manifest '$OUT/attribute_manifest.stage91-supervised.csv' \
    --output-unlabeled-manifest '$OUT/attribute_manifest.stage91-unlabeled.csv' \
    --output-report '$OUT/stage91-inatrc-train-crops-report.json' \
    >'$LOG' 2>&1"

echo "started: $SESSION"
