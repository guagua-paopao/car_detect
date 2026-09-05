#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code
RUNS=$BASE/runs/attributes
RUNNER=$CODE/training/scripts/run_stage71_final_test_after_student_validation.py
VALIDATION_STATE=$RUNS/ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2.state.json
VALIDATION_SESSION=VCAS-STAGE71-RECOVERED-CHAIN-V2
OUT=$RUNS/ATTR-STAGE71-FINAL-TEST-V1
STATE=$RUNS/ATTR-STAGE71-FINAL-TEST-V1.state.json

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "50a49cf221055bf996ccb54f4b2fa731bd545d9a445cbe643ed629d9d041c33a" ]]
test -f "$VALIDATION_STATE"
test ! -e "$OUT"
test ! -e "$STATE"

exec "$PY" "$RUNNER" \
  --validation-state "$VALIDATION_STATE" \
  --validation-session "$VALIDATION_SESSION" \
  --hard-manifest "$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv" \
  --expected-hard-manifest-sha256 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40 \
  --ua-manifest "$BASE/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv" \
  --expected-ua-manifest-sha256 6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f \
  --vfg-manifest "$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv" \
  --expected-vfg-manifest-sha256 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3 \
  --production-checkpoint "$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --expected-production-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --labels "$CODE/config/vehicle_labels.v1.json" \
  --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
  --scripts-root "$CODE/training/scripts" \
  --expected-static-evaluator-sha256 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb \
  --expected-track-evaluator-sha256 20881e01aec678e5f59f2a4eebb851cf9c015ccca5d4009780024a988459da90 \
  --output-root "$OUT" \
  --state "$STATE" \
  --python "$PY" \
  --device cuda \
  --timeout-hours 48 \
  --production-type-threshold 0.75 \
  --production-color-threshold 0.70
