#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage141_letterbox_r1"
RUNNER="$CODE/scripts/run_stage107_independent_test_once.py"
BUILDER="$CODE/scripts/build_stage107_test_views.py"
EVALUATOR="$CODE/scripts/evaluate_v2_decoupled_shared_test_once.py"
BASE_EVALUATOR="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
STAGE147_STATE="$BASE/runs/attributes/ATTR-STAGE147-INTEGRATED-COMPONENT-GATE-R1.state.json"
COLOR_REPORTS="$BASE/runs/attributes/ATTR-STAGE109-COLOR-CLASS-THRESHOLD-VALIDATION-R1"
BODY_REPORTS="$BASE/runs/attributes/ATTR-STAGE146-AXLE-LETTERBOX-MULTIRES-VALIDATION-R1"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE145-AXLE-DOMAIN-LETTERBOX-MULTIRES-R1/ATTR-STAGE145-TRUCK-SUBTYPE-CONVNEXT-288-AXLE-LETTERBOX-R1/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2/gate-best.pt"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE148-INTEGRATED-INDEPENDENT-TEST-ONCE-R1"
STATE="$OUTPUT.state.json"
LOG="$OUTPUT.log"
SESSION=VCAS-STAGE148-INTEGRATED-INDEPENDENT-TEST-ONCE-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

worker() {
  exec >"$LOG" 2>&1
  assert_sha256 "$STAGE147_STATE" 09729ab6e697568a25487ee1ad57cd6ca2ed063353b98616871568e61face9b0
  assert_sha256 "$RUNNER" 13f04e11a94322eafc23d6a15c3b701d993674336122d73c1323c18cbc7c6b58
  assert_sha256 "$BUILDER" 7d0244f085594b8238620b30e0394d3dc36a0de7dd6ecbb420fc42c7ea5217c7
  assert_sha256 "$EVALUATOR" 0372b952424f6994dea52bbbb9b82385d721f16a55fdf786905b6f1537a4421d
  assert_sha256 "$BASE_EVALUATOR" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
  assert_sha256 "$CODE/scripts/evaluate_stage109_color_class_thresholds.py" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
  assert_sha256 "$CODE/scripts/evaluate_stage110_body_class_thresholds.py" 0c7d105a54101ed2c8e8b9f803339f10ae7c46644479a36f40107b0a0a269f17
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  assert_sha256 "$SPECIALIST" 0cd4a23d8388d160cc652120aa80f64df1f051a9efd99ae97ba1f9057d317dde
  assert_sha256 "$COLOR" 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2
  assert_sha256 "$COLOR_REPORTS/gate-best/report.json" 017349da875fac3999112cf73651fd0f2f78404cfa278996d7ba25da7f3f2d8e
  assert_sha256 "$BODY_REPORTS/s288-best/report.json" 76aa9ff33f43d87930cdf18cbba95bb56e8ffec6d73d36b520bff9e2f63bd91e
  cd "$CODE"
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_evaluate_v2_decoupled_shared_test_once.py' -q
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_build_stage107_test_views.py' -q
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_run_stage107_independent_test_once.py' -q
  "$PY" "$RUNNER" \
    --stage106-state "$STAGE147_STATE" \
    --expected-stage106-state-sha256 09729ab6e697568a25487ee1ad57cd6ca2ed063353b98616871568e61face9b0 \
    --stage104-dir "$COLOR_REPORTS" --stage105-dir "$BODY_REPORTS" \
    --test-view-builder "$BUILDER" --expected-test-view-builder-sha256 7d0244f085594b8238620b30e0394d3dc36a0de7dd6ecbb420fc42c7ea5217c7 \
    --fixed-test-evaluator "$EVALUATOR" --expected-fixed-test-evaluator-sha256 0372b952424f6994dea52bbbb9b82385d721f16a55fdf786905b6f1537a4421d \
    --base-evaluator "$BASE_EVALUATOR" --expected-base-evaluator-sha256 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12 \
    --hard-manifest "$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv" --expected-hard-manifest-sha256 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40 \
    --ua-manifest "$BASE/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv" --expected-ua-manifest-sha256 6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f \
    --vfg-manifest "$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv" --expected-vfg-manifest-sha256 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3 \
    --labels "$BASE/code/config/vehicle_labels.v2.json" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --baseline-checkpoint "$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
    --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 0cd4a23d8388d160cc652120aa80f64df1f051a9efd99ae97ba1f9057d317dde \
    --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2 \
    --datasets-safety-root "$BASE/datasets" --output-root "$OUTPUT" --state "$STATE" \
    --python "$PY" --device cuda --batch-size 64 --workers 8
  sha256sum "$STATE" >"$STATE.sha256"
  if [[ -f "$OUTPUT/stage107-independent-test-report.json" ]]; then
    sha256sum "$OUTPUT/stage107-independent-test-report.json" >"$OUTPUT/stage107-independent-test-report.json.sha256"
  fi
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
for absent in "$OUTPUT" "$STATE" "$STATE.sha256" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage148 evidence: $absent" >&2; exit 66; }
done
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker"
echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
