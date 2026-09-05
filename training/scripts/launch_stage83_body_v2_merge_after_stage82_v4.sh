#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
OUT="$ROOT/datasets/attribute-domain-v2/stage83-body-v2-merged-v4"
STATE="$ROOT/runs/attributes/ATTR-STAGE83-BODY-V2-MERGE-V4.state.json"
LOG="$ROOT/runs/attributes/ATTR-STAGE83-BODY-V2-MERGE-V4.log"
SESSION=VCAS-STAGE83-BODY-V2-MERGE-V4

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

tmux new-session -d -s "$SESSION" \
  "/root/miniconda3/bin/python '$ROOT/code/training/scripts/run_stage83_body_v2_merge_after_stage82.py' \
    --stage82-report '$ROOT/datasets/attribute-domain-v2/stage82-mio-train-only-v1/stage82-mio-train-only-report.json' \
    --stage82-manifest '$ROOT/datasets/attribute-domain-v2/stage82-mio-train-only-v1/attribute_manifest.stage82-mio-train-only.csv' \
    --base-manifest '$ROOT/datasets/attribute-domain-v2/stage72-body-taxonomy-v2-v1/attribute_manifest.stage72-body-taxonomy-v2.csv' \
    --expected-base-sha256 0b28f8bab925c5e5c6b680af1fcb6d35a4a76a09d7ca1f47ec0f02e8c74a4aed \
    --labels '$ROOT/code/config/vehicle_labels.v2.json' \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --merger '$ROOT/code/training/scripts/build_stage83_body_v2_merged_manifest.py' \
    --expected-merger-sha256 411fc02de6374b289f1910c08c8e7aaf54ebf7147ad83bdff7b2b1f3a3e3109b \
    --output-manifest '$OUT/attribute_manifest.stage83-body-v2-merged.csv' \
    --output-report '$OUT/stage83-body-v2-merged-report.json' \
    --state '$STATE' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
