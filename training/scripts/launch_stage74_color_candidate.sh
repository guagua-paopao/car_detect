#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
RUNNER="$CODE_ROOT/training/scripts/run_stage74_color_candidate.py"
MATRIX="$CODE_ROOT/training/stage74-color-candidate-matrix.json"
OUTPUT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1
STATE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1.state.json
LOG=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1.launch.log
PREFLIGHT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1.preflight-output
PREFLIGHT_STATE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1.preflight-state.json
PREFLIGHT_REPORT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V1.preflight.json
SESSION=VCAS-STAGE74-COLOR-TRAIN

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print toupper($1)}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  fi
}

for required in "$PYTHON" "$RUNNER" "$MATRIX"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$RUNNER" F2975D4FA1BDABFC3FCABA51A4755A824ABF366504AE7CAEE1C4C4F1CB1F2FAD
assert_sha256 "$MATRIX" 92E095EDAEADC866F14B2ED0CDBD55BE61953CD3E6B8C81217306E6F77C02F23

for output in "$OUTPUT_ROOT" "$STATE" "$LOG" "$PREFLIGHT_ROOT" "$PREFLIGHT_STATE" "$PREFLIGHT_REPORT"; do
  [[ ! -e "$output" ]] || { echo "refusing to overwrite Stage74 evidence: $output" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
if pgrep -f '^/root/miniconda3/bin/python .*/train_attribute.py.*ATTR-STAGE74-COLOR' >/dev/null \
  || pgrep -f '^/root/miniconda3/bin/python .*/run_stage74_color_candidate.py' >/dev/null; then
  echo "a Stage74 color training process is already active" >&2
  exit 68
fi

"$PYTHON" "$RUNNER" \
  --matrix "$MATRIX" \
  --output-root "$PREFLIGHT_ROOT" \
  --state "$PREFLIGHT_STATE" \
  --device cuda \
  --code-revision stage74-color-cctv-joint-research-v1 \
  --preflight-only >"$PREFLIGHT_REPORT"

grep -q '"status": "pass_preflight_only"' "$PREFLIGHT_REPORT" || {
  echo "Stage74 preflight did not pass" >&2
  exit 69
}

tmux new-session -d -s "$SESSION" \
  "$PYTHON '$RUNNER' \
    --matrix '$MATRIX' \
    --output-root '$OUTPUT_ROOT' \
    --state '$STATE' \
    --device cuda \
    --code-revision stage74-color-cctv-joint-research-v1 \
    > '$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "preflight=$PREFLIGHT_REPORT"
echo "state=$STATE"
echo "log=$LOG"
