#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code
RUNS=$BASE/runs/attributes
RUNNER=$CODE/training/scripts/run_stage71_onnx_after_final_test_v2.py
OUT=$RUNS/ATTR-STAGE71-BACKEND-CANDIDATE-V2
STATE=$RUNS/ATTR-STAGE71-BACKEND-CANDIDATE-V2.state.json

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "e1fa276cc5d345c9cc224dd53e1b96afe5b5db8e99ec203f00a404d6b37d3171" ]]
test ! -e "$OUT"
test ! -e "$STATE"

exec "$PY" "$RUNNER" \
  --final-test-state "$RUNS/ATTR-STAGE71-FINAL-TEST-V1.state.json" \
  --final-test-session VCAS-STAGE71-FINAL-TEST \
  --scripts-root "$CODE/training/scripts" \
  --expected-exporter-sha256 5efd3246bd2ca35c0b58668086548f2245c133c46cfb7f0b4b29d0673beb303c \
  --expected-parity-validator-sha256 bebe67077ab93dc148a8d525311fbc9177c525bde181af52511b853ebb7e768f \
  --expected-trt-input-preparer-sha256 38a9ed6c077bcf4b80f41de946084e5ad6358f57904e15bf107c2ca0fd640ee9 \
  --labels "$CODE/config/vehicle_labels.v1.json" \
  --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
  --validation-manifest "$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv" \
  --expected-validation-manifest-sha256 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40 \
  --output-root "$OUT" \
  --state "$STATE" \
  --python "$PY" \
  --device cuda \
  --timeout-hours 48
