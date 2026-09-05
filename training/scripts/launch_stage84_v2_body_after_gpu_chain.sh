#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage84_v2_body_after_gpu_chain.py"
MATRIX="$CODE/training/stage84-v2-body-candidate-matrix-v1.json"
STAGE76_STATE="$BASE/runs/attributes/ATTR-STAGE76-BODY-PRETRAIN-FINETUNE-V1.state.json"
STAGE80_STATE="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM.state.json"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-V1"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.runner.log"
PREFLIGHT="$OUTPUT_ROOT.preflight.log"
SESSION=VCAS-STAGE84-BODY-V2

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

for required in "$PY" "$RUNNER" "$MATRIX" "$STAGE76_STATE" "$STAGE80_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$RUNNER" 1df398bc834fa61a2d7d73023b6a956027958788c55ac3d3fe6fab58820f2ea3
assert_sha256 "$MATRIX" 75dca7d5d53325b0e3a6ad26132e911c819bc6f8ac24e5a82a2fbb3d3703e713

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
  --stage76-state "$STAGE76_STATE" --stage80-state "$STAGE80_STATE" \
  --output-root "$OUTPUT_ROOT" --state "$STATE" \
  --device cuda --code-revision stage84-v2-body-mio-cctv-v1 \
  --preflight-only >"$PREFLIGHT" 2>&1

tmux new-session -d -s "$SESSION" \
  "'$PY' '$RUNNER' \
    --matrix '$MATRIX' \
    --stage76-state '$STAGE76_STATE' --stage80-state '$STAGE80_STATE' \
    --output-root '$OUTPUT_ROOT' --state '$STATE' \
    --device cuda --code-revision stage84-v2-body-mio-cctv-v1 \
    --poll-seconds 30 --timeout-seconds 172800 >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
