#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage85_v2_decoupled_shared_validation.py"
MATRIX="$CODE/training/stage85-v2-decoupled-hierarchical-validation-matrix-r3.json"
STAGE80_STATE="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3.state.json"
STAGE84_STATE="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2.state.json"
DATASETS_ROOT="$BASE/datasets"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE85-V2-DECOUPLED-HIERARCHICAL-VALIDATION-R3"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.runner.log"
PREFLIGHT="$OUTPUT_ROOT.preflight.log"
SESSION=VCAS-STAGE85-V2-HIER-VAL-R3

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

for required in "$PY" "$RUNNER" "$MATRIX" "$STAGE80_STATE" "$STAGE84_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
[[ -d "$DATASETS_ROOT" ]] || { echo "missing datasets root: $DATASETS_ROOT" >&2; exit 65; }
assert_sha256 "$RUNNER" e87df7b8d2e2ac63ebccd6f5d71d3ffaf7322c60864e367e95cd3968fb1137c9
assert_sha256 "$MATRIX" 4af3f02d9e393cc7cc6e4bd7c3fa311f7d1d787676314daa4713eb09c4138680

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
[[ ! -e "$PREFLIGHT" ]] || { echo "refusing to overwrite: $PREFLIGHT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" "$RUNNER" \
  --matrix "$MATRIX" \
  --stage80-state "$STAGE80_STATE" --stage84-state "$STAGE84_STATE" \
  --output-root "$OUTPUT_ROOT" --state "$STATE" \
  --datasets-safety-root "$DATASETS_ROOT" --device cuda \
  --preflight-only >"$PREFLIGHT" 2>&1

tmux new-session -d -s "$SESSION" \
  "'$PY' '$RUNNER' \
    --matrix '$MATRIX' \
    --stage80-state '$STAGE80_STATE' --stage84-state '$STAGE84_STATE' \
    --output-root '$OUTPUT_ROOT' --state '$STATE' \
    --datasets-safety-root '$DATASETS_ROOT' --device cuda \
    --batch-size 64 --workers 8 --poll-seconds 30 --timeout-seconds 259200 \
    >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
