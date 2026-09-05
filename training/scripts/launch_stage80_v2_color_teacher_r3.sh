#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage80_v2_color_teacher_after_stage77.py"
MATRIX="$CODE/training/stage80-v2-color-teacher-matrix-r3.json"
STAGE77_STATE="$BASE/runs/attributes/ATTR-STAGE77-BODY-VALIDATION-V1.state.json"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.runner.log"
PREFLIGHT="$OUTPUT_ROOT.preflight.log"
SESSION=VCAS-STAGE80-COLOR-V2-TEACHER-R3

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

for required in "$PY" "$RUNNER" "$MATRIX" "$STAGE77_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$RUNNER" eadf0186d1560cd8e1aebf5da8cf3fae1c6f535b241d01a05af34046d9108aca
assert_sha256 "$MATRIX" f6f3ee9c68f3d33e2ae5b9f818c17ad2559627128d2fe9a7ec78cf2875391418

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
[[ ! -e "$PREFLIGHT" ]] || { echo "refusing to overwrite: $PREFLIGHT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" "$RUNNER" \
  --matrix "$MATRIX" --stage77-state "$STAGE77_STATE" \
  --output-root "$OUTPUT_ROOT" --state "$STATE" \
  --device cuda --code-revision stage80-v2-color-teacher-isolated-r3 \
  --preflight-only >"$PREFLIGHT" 2>&1

tmux new-session -d -s "$SESSION" \
  "'$PY' '$RUNNER' \
    --matrix '$MATRIX' --stage77-state '$STAGE77_STATE' \
    --output-root '$OUTPUT_ROOT' --state '$STATE' \
    --device cuda --code-revision stage80-v2-color-teacher-isolated-r3 \
    --poll-seconds 30 --timeout-seconds 21600 >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
