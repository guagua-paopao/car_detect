#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage102_partial_color_r1"
DECIDER="$CODE/scripts/decide_stage106_integrated_candidate.py"
COLOR_STATE="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R2.state.json"
BODY_STATE="$BASE/runs/attributes/ATTR-STAGE105-BODY-SUBTYPE-THRESHOLD-SWEEP-R1.state.json"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE106-INTEGRATED-COMPONENT-GATE-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE106-INTEGRATED-COMPONENT-GATE-R1.log"
SESSION=VCAS-STAGE106-INTEGRATED-COMPONENT-GATE-R1
UPSTREAM_SESSION=VCAS-STAGE105-BODY-SUBTYPE-SWEEP-R1
SELF="$CODE/scripts/launch_stage106_integrated_component_gate_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 30
  done
  [[ -f "$COLOR_STATE" ]] || { echo "missing Stage104 state: $COLOR_STATE" >&2; exit 70; }
  [[ -f "$BODY_STATE" ]] || { echo "missing Stage105 state: $BODY_STATE" >&2; exit 71; }
  "$PY" "$DECIDER" \
    --color-state "$COLOR_STATE" \
    --body-state "$BODY_STATE" \
    --output "$OUTPUT"
  sha256sum "$OUTPUT" > "$OUTPUT.sha256"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$OUTPUT.sha256" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage106 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "75205c07ac52c7bfb9fa2568091ee9c5baeb623c53b98f236b6766e914cd92e5  $DECIDER" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$OUTPUT"
echo "log=$LOG"
