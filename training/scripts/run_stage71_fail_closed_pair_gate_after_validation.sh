#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code
RUNS=$BASE/runs/attributes
RUNNER=$CODE/training/scripts/run_stage71_fail_closed_pair_gate.py
UPSTREAM_STATE=$RUNS/ATTR-STAGE71-STUDENT-PAIR-VALIDATION-ONLY-V1.state.json
OUT=$RUNS/ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2
STATE=$RUNS/ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2.state.json

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "2c3bd7d3731d2c821295c2736c23400e2c6fdfb16aad6be10557d709d3158422" ]]
[[ "$(sha256sum "$CODE/training/scripts/sweep_attribute_track_fusion_fail_closed_v2.py" | awk '{print $1}')" == "e3dcfa41d3d2e0a787eae77824309c6cca5641c8dd413ac55136dd585da28e5d" ]]
[[ "$(sha256sum "$CODE/training/scripts/track_fusion_fail_closed_v2.py" | awk '{print $1}')" == "faca95f8b17661a4992b566e471def9b7f09ac3914741a88612e8fecfb396264" ]]
test -f "$UPSTREAM_STATE"
test ! -e "$OUT"
test ! -e "$STATE"

exec "$PY" "$RUNNER" \
  --upstream-state "$UPSTREAM_STATE" \
  --ua-manifest "$BASE/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv" \
  --expected-ua-manifest-sha256 6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f \
  --vfg-manifest "$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv" \
  --expected-vfg-manifest-sha256 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3 \
  --production-checkpoint "$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --expected-production-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --scripts-root "$CODE/training/scripts" \
  --expected-v1-sweeper-sha256 e3310eda88ae9e8afec5337408fd5fe9f977a0a176ac1b8f55375722c195e502 \
  --expected-v1-evaluator-sha256 20881e01aec678e5f59f2a4eebb851cf9c015ccca5d4009780024a988459da90 \
  --expected-v2-fusion-sha256 faca95f8b17661a4992b566e471def9b7f09ac3914741a88612e8fecfb396264 \
  --expected-v2-sweeper-sha256 e3dcfa41d3d2e0a787eae77824309c6cca5641c8dd413ac55136dd585da28e5d \
  --output-root "$OUT" \
  --state "$STATE" \
  --python "$PY" \
  --device cuda
