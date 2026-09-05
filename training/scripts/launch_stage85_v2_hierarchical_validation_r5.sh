#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/vcas/code/training_stage85_r4
EVALUATOR=/root/autodl-tmp/vcas/code/training_stage84_v2/scripts/evaluate_v2_decoupled_hierarchical_validation_r4.py
TEST_EVALUATOR=$ROOT/scripts/evaluate_v2_decoupled_shared_validation.py
TEST=$ROOT/tests/test_evaluate_v2_decoupled_shared_validation.py
RUNNER=$ROOT/run_stage85_v2_decoupled_shared_validation.py
MATRIX=$ROOT/stage85-v2-decoupled-hierarchical-validation-matrix-r4.json
STAGE80=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3.state.json
STAGE84=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2.state.json
OUTPUT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE85-V2-DECOUPLED-HIERARCHICAL-VALIDATION-R5
STATE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE85-V2-DECOUPLED-HIERARCHICAL-VALIDATION-R5.state.json

test ! -e "$OUTPUT"
test ! -e "$STATE"
echo "43f90106fb4754bae9c6e0123ff38060bdd11c46b33e442cb49f7aa93179abf3  $EVALUATOR" | sha256sum -c -
echo "43f90106fb4754bae9c6e0123ff38060bdd11c46b33e442cb49f7aa93179abf3  $TEST_EVALUATOR" | sha256sum -c -
echo "07449ba1bc627fdf6deca859954337620393b31a3ddaf6b200d66136a4999152  $TEST" | sha256sum -c -
echo "e87df7b8d2e2ac63ebccd6f5d71d3ffaf7322c60864e367e95cd3968fb1137c9  $RUNNER" | sha256sum -c -
echo "37f2a7c3953fce908a190693efee18efb1ff0c5dce81288120277ea77de71505  $MATRIX" | sha256sum -c -

cd "$ROOT"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_evaluate_v2_decoupled_shared_validation.py'
"$PYTHON" -c "import importlib.util; p='$EVALUATOR'; s=importlib.util.spec_from_file_location('stage85_r4_eval', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); from src.attribute_hierarchy import decode_truck_family; print(m.TRAINING_ROOT, decode_truck_family.__name__)"

COMMON=(
  --matrix "$MATRIX"
  --stage80-state "$STAGE80"
  --stage84-state "$STAGE84"
  --output-root "$OUTPUT"
  --state "$STATE"
  --datasets-safety-root /root/autodl-tmp/vcas/datasets
  --device cuda
  --batch-size 64
  --workers 8
)
"$PYTHON" "$RUNNER" "${COMMON[@]}" --preflight-only
exec "$PYTHON" "$RUNNER" "${COMMON[@]}"
