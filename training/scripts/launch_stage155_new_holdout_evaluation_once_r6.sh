#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage155_eval_r1"
EVALUATOR="$CODE/scripts/evaluate_stage155_new_holdout_once.py"
EVAL_BASE="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
INTEGRATED_STATE="$BASE/runs/attributes/ATTR-STAGE154-INTEGRATED-COMPONENT-GATE-R2.state.json"
HOLDOUT_STATE="$BASE/datasets/attribute-domain-v2/stage155-new-unused-holdout-r1.state.json"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
EVAL_ROOT="$BASE/runs/attributes/ATTR-STAGE155-NEW-UNUSED-HOLDOUT-R1"
REPORT="$EVAL_ROOT/report.json"
STATE="$EVAL_ROOT.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

[[ ! -e "$REPORT" && ! -e "$STATE" && ! -e "$REPORT.sha256" && ! -e "$STATE.sha256" ]] || {
  echo "refusing to overwrite Stage155 one-time evaluation evidence" >&2
  exit 66
}

assert_sha256 "$EVALUATOR" 8dc0097d63d5bf9f3ed9b59ed13b02e6f764e4ba8812b4aeb6b1846dd2f45d81
assert_sha256 "$EVAL_BASE" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$INTEGRATED_STATE" 7e09040213d59f14fa72f2462c712721a7121a0c2535661fab4db2950ae84b44
assert_sha256 "$HOLDOUT_STATE" 43b55a6527ff48cd28d41d2b442a95ee0c0aad4f84f07e1ca0e08304150d741d
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
"$PY" -m py_compile "$EVALUATOR" "$EVAL_BASE"

mkdir -p "$EVAL_ROOT"
set +e
"$PY" "$EVALUATOR" \
  --integrated-state "$INTEGRATED_STATE" --expected-integrated-state-sha256 7e09040213d59f14fa72f2462c712721a7121a0c2535661fab4db2950ae84b44 \
  --holdout-state "$HOLDOUT_STATE" --expected-holdout-state-sha256 43b55a6527ff48cd28d41d2b442a95ee0c0aad4f84f07e1ca0e08304150d741d \
  --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --baseline-body-threshold 0.978 --baseline-color-threshold 0.742 \
  --body-specialist-subtype-threshold 0.50 --datasets-safety-root "$BASE/datasets" \
  --device cuda --batch-size 64 --workers 8 --output "$REPORT" --state-output "$STATE"
code=$?
set -e
[[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
exit "$code"
