#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
EVALUATOR="$BASE/code/stage159_component_validation_r6/scripts/evaluate_stage159_component_validation.py"
FINALIZER="$BASE/code/stage244-color-validation-r1/finalize_stage244_color_validation.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage159-validation-views-r3/color.validation-only.csv"
CONFIG="$BASE/code/stage159_component_validation_r6/production_vehicle_analytics.yaml"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
REFERENCE="$BASE/runs/attributes/ATTR-STAGE159-COMPONENT-VALIDATION-R8.state.json"
TRAINING_STATE="$BASE/runs/attributes/ATTR-STAGE243-NIGHTOWLS-COLOR-REPAIR-R1/state.json"
TRAINING_SESSION=VCAS-STAGE243-NIGHTOWLS-COLOR-REPAIR-R1
ROOT="$BASE/runs/attributes/ATTR-STAGE244-COLOR-VALIDATION-R2"
STATE="$ROOT/state.json"
LOG="$BASE/logs/stage244-color-validation-r2.log"
SESSION=VCAS-STAGE244-COLOR-VALIDATION-R2

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$EVALUATOR" e8147f470e7a3d48f46d67c4b41f32bc7963c7248da1a0f640f069a9ed037e1d
assert_sha256 "$FINALIZER" be60fa7f06842979913532fbecbe38db2b6e7f053fb9909359e081c63a42e41c
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" 34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023
assert_sha256 "$CONFIG" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$REFERENCE" 87cfd43df79fd32243d06ed954d4b36aebc3724eef40f3a0ebf20d94d27a8d92
"$PY" -m py_compile "$EVALUATOR" "$FINALIZER"

if [[ "${STAGE244_WORKER:-0}" != "1" ]]; then
  test ! -e "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION" >&2
    exit 67
  fi
  tmux new-session -d -s "$SESSION" \
    "STAGE244_WORKER=1 bash '$BASE/code/stage244-color-validation-r2/launch_stage244_color_validation_r1.sh'"
  echo "started tmux:$SESSION"
  echo "root=$ROOT"
  echo "log=$LOG"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "Stage244 waiting for Stage243 completion"
while tmux has-session -t "$TRAINING_SESSION" 2>/dev/null || \
      pgrep -af 'train_attribute.py.*ATTR-STAGE243' >/dev/null 2>&1; do
  sleep 20
done

test -f "$TRAINING_STATE"
test -f "$TRAINING_STATE.sha256"
expected_state_sha="$(awk 'NR==1 {print tolower($1)}' "$TRAINING_STATE.sha256")"
assert_sha256 "$TRAINING_STATE" "$expected_state_sha"

mapfile -t candidate_lines < <("$PY" - "$TRAINING_STATE" <<'PY'
import json, sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert state['status'] == 'training_complete_pending_stage244_validation'
assert state['test_accessed'] is False
assert state['frozen_video_used'] is False
assert state['production_model_modified'] is False
assert state['deployment_performed'] is False
for key in ('night_weight_12', 'night_weight_20'):
    artifact = state['candidates'][key]['artifacts']['best.pt']
    print(f"{key}\t{artifact['path']}\t{artifact['sha256']}")
PY
)
test "${#candidate_lines[@]}" -eq 2
test ! -e "$ROOT"
mkdir -p "$ROOT"

for row in "${candidate_lines[@]}"; do
  IFS=$'\t' read -r variant checkpoint checkpoint_sha <<<"$row"
  assert_sha256 "$checkpoint" "$checkpoint_sha"
  output="$ROOT/$variant/report.json"
  echo "Stage244 validating $variant"
  set +e
  "$PY" "$EVALUATOR" \
    --head color \
    --manifest "$MANIFEST" \
    --expected-manifest-sha256 34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023 \
    --labels "$LABELS" \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --candidate-checkpoint "$checkpoint" \
    --expected-candidate-sha256 "$checkpoint_sha" \
    --baseline-checkpoint "$BASELINE" \
    --expected-baseline-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --production-config "$CONFIG" \
    --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
    --resolution-root "$BASE" \
    --allowed-image-root "$BASE/datasets" \
    --allowed-image-root "$BASE/sources" \
    --precision-target 0.935 \
    --device cuda --batch-size 64 --workers 8 \
    --output "$output"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 && "$rc" -ne 2 ]]; then
    echo "unexpected evaluator exit code for $variant: $rc" >&2
    exit "$rc"
  fi
done

"$PY" "$FINALIZER" \
  --validation-root "$ROOT" \
  --reference-state "$REFERENCE" \
  --expected-reference-sha256 87cfd43df79fd32243d06ed954d4b36aebc3724eef40f3a0ebf20d94d27a8d92 \
  --training-state "$TRAINING_STATE" \
  --output "$STATE"

find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
echo "Stage244 complete: $STATE"
