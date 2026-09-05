#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code
RUNNER=$CODE/training/scripts/run_stage71_teacher_pair_validation_after_training.py
TRAINING_OUT=$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1
OUT=$BASE/runs/attributes/ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1
STATE=$BASE/runs/attributes/ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1.state.json

test ! -e "$OUT"
test ! -e "$STATE"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "22e3f57658b714c5e4f4134bdb8eb6caf5fcd428b6819f0855b0235b6d03972d" ]]

exec "$PY" "$RUNNER" \
  --training-state "$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1.state.json" \
  --training-matrix "$CODE/training/stage71-convnext-teacher-matrix-v1.json" \
  --expected-training-matrix-sha256 b6909be8f082147e973fba8cf9e6446d49e59b782a382cfee8788c604abab66f \
  --training-root "$TRAINING_OUT" \
  --training-session VCAS-TRAIN-STAGE71-TEACHERS-V1 \
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
  --expected-track-sweeper-sha256 e3310eda88ae9e8afec5337408fd5fe9f977a0a176ac1b8f55375722c195e502 \
  --expected-vfg-comparison-sha256 5a924c27bc927a82c79c158341e4c3188333edd76d07f4b4cf84a050a19a6899 \
  --expected-track-evaluator-sha256 20881e01aec678e5f59f2a4eebb851cf9c015ccca5d4009780024a988459da90 \
  --output-root "$OUT" \
  --state "$STATE" \
  --python "$PY" \
  --device cuda \
  --timeout-hours 10 \
  --expected-candidates 2 \
  --schema-prefix stage71-convnext-teacher \
  --passing-decision "passing teacher pairs may proceed only to conservative train-pool pseudo-label mining and student distillation; no test, backend, frozen-video replay or deployment" \
  --failure-decision "reject every teacher pair before pseudo-label mining, student distillation, test, backend, frozen-video replay or deployment"
