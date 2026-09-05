#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
RUNNER="$CODE/scripts/run_stage107_independent_test_once.py"
BUILDER="$CODE/scripts/build_stage107_test_views.py"
EVALUATOR="$CODE/scripts/evaluate_v2_decoupled_shared_test_once.py"
BASE_EVALUATOR="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
STAGE106_STATE="$BASE/runs/attributes/ATTR-STAGE106-INTEGRATED-COMPONENT-GATE-R1.state.json"
STAGE104_DIR="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R2"
STAGE105_DIR="$BASE/runs/attributes/ATTR-STAGE105-BODY-SUBTYPE-THRESHOLD-SWEEP-R1"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1.log"
SESSION=VCAS-STAGE107-INTEGRATED-INDEPENDENT-TEST-ONCE-R1
UPSTREAM_SESSION=VCAS-STAGE106-INTEGRATED-COMPONENT-GATE-R1
SELF="$CODE/scripts/launch_stage107_independent_test_once_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 30
  done
  [[ -f "$STAGE106_STATE" ]] || { echo "missing Stage106 state" >&2; exit 70; }
  [[ -f "$STAGE106_STATE.sha256" ]] || { echo "missing Stage106 state sidecar" >&2; exit 71; }
  sha256sum -c "$STAGE106_STATE.sha256"
  local stage106_sha
  stage106_sha="$(sha256sum "$STAGE106_STATE" | awk '{print tolower($1)}')"
  "$PY" "$RUNNER" \
    --stage106-state "$STAGE106_STATE" \
    --expected-stage106-state-sha256 "$stage106_sha" \
    --stage104-dir "$STAGE104_DIR" \
    --stage105-dir "$STAGE105_DIR" \
    --test-view-builder "$BUILDER" \
    --expected-test-view-builder-sha256 7d0244f085594b8238620b30e0394d3dc36a0de7dd6ecbb420fc42c7ea5217c7 \
    --fixed-test-evaluator "$EVALUATOR" \
    --expected-fixed-test-evaluator-sha256 0beea119cf6d41dbf681b302a4836d839302834f36e1e9c872ee7515ada541de \
    --base-evaluator "$BASE_EVALUATOR" \
    --expected-base-evaluator-sha256 447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38 \
    --hard-manifest "$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv" \
    --expected-hard-manifest-sha256 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40 \
    --ua-manifest "$BASE/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv" \
    --expected-ua-manifest-sha256 6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f \
    --vfg-manifest "$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv" \
    --expected-vfg-manifest-sha256 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3 \
    --labels "$BASE/code/config/vehicle_labels.v2.json" \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --baseline-checkpoint "$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
    --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --datasets-safety-root "$BASE/datasets" \
    --output-root "$OUTPUT" \
    --state "$STATE" \
    --python "$PY" \
    --device cuda \
    --batch-size 64 \
    --workers 8
  sha256sum "$STATE" > "$STATE.sha256"
  if [[ -f "$OUTPUT/stage107-independent-test-report.json" ]]; then
    sha256sum "$OUTPUT/stage107-independent-test-report.json" > \
      "$OUTPUT/stage107-independent-test-report.json.sha256"
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$STATE.sha256" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage107 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "286b232e0d653b3dd9fbaf2b196d2e99e403c25e8649e760d726f99134793bcc  $RUNNER" | sha256sum -c -
echo "7d0244f085594b8238620b30e0394d3dc36a0de7dd6ecbb420fc42c7ea5217c7  $BUILDER" | sha256sum -c -
echo "0beea119cf6d41dbf681b302a4836d839302834f36e1e9c872ee7515ada541de  $EVALUATOR" | sha256sum -c -
echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $BASE_EVALUATOR" | sha256sum -c -
echo "51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40  $BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv" | sha256sum -c -
echo "6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f  $BASE/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv" | sha256sum -c -
echo "5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3  $BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
