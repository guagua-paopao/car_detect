#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
DECIDER="$CODE/scripts/decide_stage116_threshold_integrated_candidate.py"
COLOR_STATE="$BASE/runs/attributes/ATTR-STAGE109-COLOR-CLASS-THRESHOLD-VALIDATION-R1.state.json"
BODY_STATE="$BASE/runs/attributes/ATTR-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R2.state.json"
BODY_SESSION=VCAS-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R2
STATE="$BASE/runs/attributes/ATTR-STAGE116-THRESHOLD-INTEGRATED-COMPONENT-GATE-R2.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE116-THRESHOLD-INTEGRATED-COMPONENT-GATE-R2.log"
SESSION=VCAS-STAGE116-THRESHOLD-INTEGRATED-GATE-R2

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

worker() {
  while tmux has-session -t "$BODY_SESSION" 2>/dev/null; do sleep 30; done
  for required in "$PY" "$DECIDER" "$COLOR_STATE" "$BODY_STATE"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$DECIDER" 7d41d136169ffd77afeabcca615f4925e6c09b5fb3d1885651c076d991c30e27
  assert_sha256 "$COLOR_STATE" 5ab047025832e0cf511b993ab3468288e8a2040756dac9e8d28756516c7252a3
  body_sha="$(sha256sum "$BODY_STATE" | awk '{print tolower($1)}')"
  [[ ! -e "$STATE" ]] || { echo "refusing to overwrite Stage116 R2 state" >&2; exit 66; }
  set +e
  "$PY" "$DECIDER" \
    --color-state "$COLOR_STATE" --expected-color-state-sha256 5ab047025832e0cf511b993ab3468288e8a2040756dac9e8d28756516c7252a3 \
    --body-state "$BODY_STATE" --expected-body-state-sha256 "$body_sha" \
    --output "$STATE"
  code=$?
  set -e
  [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$STATE" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage116 R2" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
