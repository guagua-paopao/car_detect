#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage86_secondary_after_stage85.py"
MATRIX="$CODE/training/stage86-secondary-body-validation-matrix-v1.json"
PRIMARY_STATE="$BASE/runs/attributes/ATTR-STAGE85-V2-DECOUPLED-HIERARCHICAL-VALIDATION-R3.state.json"
DATASETS_ROOT="$BASE/datasets"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE86-SECONDARY-BODY-VALIDATION-V1"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.runner.log"
PREFLIGHT="$OUTPUT_ROOT.preflight.log"
SESSION=VCAS-STAGE86-SECONDARY-BODY-VAL-V1

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

for required in "$PY" "$RUNNER" "$MATRIX" "$PRIMARY_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
[[ -d "$DATASETS_ROOT" ]] || { echo "missing datasets root: $DATASETS_ROOT" >&2; exit 65; }
assert_sha256 "$RUNNER" 2d429d8369ba842adfda9904b5896cd6bf93e8a44b773ec3edabe8703fcae136
assert_sha256 "$MATRIX" c921fda312697be14db75bfc3ad93fc285cceeb94eb76173e5fb4e076ca7a00d

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
[[ ! -e "$PREFLIGHT" ]] || { echo "refusing to overwrite: $PREFLIGHT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" "$RUNNER" \
  --matrix "$MATRIX" --primary-state "$PRIMARY_STATE" \
  --output-root "$OUTPUT_ROOT" --state "$STATE" \
  --datasets-safety-root "$DATASETS_ROOT" --device cuda \
  --preflight-only >"$PREFLIGHT" 2>&1

tmux new-session -d -s "$SESSION" \
  "'$PY' '$RUNNER' \
    --matrix '$MATRIX' --primary-state '$PRIMARY_STATE' \
    --output-root '$OUTPUT_ROOT' --state '$STATE' \
    --datasets-safety-root '$DATASETS_ROOT' --device cuda \
    --batch-size 64 --workers 8 --poll-seconds 30 --timeout-seconds 259200 \
    >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
