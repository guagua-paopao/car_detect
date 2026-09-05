#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
RUNNER="$CODE/scripts/run_stage108_three_branch_onnx_chain.py"
EXPORTER="$CODE/scripts/export_stage108_three_branch_onnx.py"
PARITY="$CODE/scripts/validate_stage108_three_branch_onnx_parity.py"
STAGE107_STATE="$BASE/runs/attributes/ATTR-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1.state.json"
VALIDATION_MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE108-THREE-BRANCH-ONNX-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE108-THREE-BRANCH-ONNX-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE108-THREE-BRANCH-ONNX-R1.log"
SESSION=VCAS-STAGE108-THREE-BRANCH-ONNX-R1
UPSTREAM_SESSION=VCAS-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1
SELF="$CODE/scripts/launch_stage108_three_branch_onnx_chain_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 30
  done
  [[ -f "$STAGE107_STATE" ]] || { echo "missing Stage107 state" >&2; exit 70; }
  [[ -f "$STAGE107_STATE.sha256" ]] || { echo "missing Stage107 state sidecar" >&2; exit 71; }
  sha256sum -c "$STAGE107_STATE.sha256"
  local stage107_sha
  stage107_sha="$(sha256sum "$STAGE107_STATE" | awk '{print tolower($1)}')"
  "$PY" "$RUNNER" \
    --stage107-state "$STAGE107_STATE" \
    --expected-stage107-state-sha256 "$stage107_sha" \
    --exporter "$EXPORTER" \
    --expected-exporter-sha256 313e3a496be8bd8662cad9423a0cfb044d0ce4b3cbecd62a75edcaf793662d03 \
    --parity-validator "$PARITY" \
    --expected-parity-validator-sha256 ed5f1ba857068e4a7da878265a4f3f07c7961d646fffdf4d61981d3625491243 \
    --labels "$LABELS" \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --validation-manifest "$VALIDATION_MANIFEST" \
    --expected-validation-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
    --datasets-safety-root "$BASE/datasets" \
    --output-root "$OUTPUT" \
    --state "$STATE" \
    --python "$PY" \
    --device cuda
  sha256sum "$STATE" > "$STATE.sha256"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$STATE.sha256" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage108 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "e68b12a5d42974dbb869aacd099a9e5e28b4b1a68527ef1162565e7af019f319  $RUNNER" | sha256sum -c -
echo "313e3a496be8bd8662cad9423a0cfb044d0ce4b3cbecd62a75edcaf793662d03  $EXPORTER" | sha256sum -c -
echo "ed5f1ba857068e4a7da878265a4f3f07c7961d646fffdf4d61981d3625491243  $PARITY" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $VALIDATION_MANIFEST" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
