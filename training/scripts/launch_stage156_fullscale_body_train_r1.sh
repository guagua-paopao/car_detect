#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
TRAINER="$BASE/code/training_stage141_letterbox_r1/scripts/train_attribute.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST_STATE="$BASE/datasets/attribute-domain-v2/stage156-fullscale-body-repair-r1.state.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage156-fullscale-body-repair-r1/attribute_manifest.stage156-fullscale-body-repair.csv"
INIT="$BASE/runs/attributes/ATTR-STAGE153-BODY-NIGHT-SMALL-REPAIR-R2/ATTR-STAGE153-BODY-CONVNEXT-256-LVAD-REPAIR-R2/last.pt"
CODE="$BASE/code/training_stage156_fullscale_body_r1"
FINALIZER="$CODE/scripts/finalize_stage156_fullscale_body_training.py"
FINALIZER_TEST="$CODE/scripts/test_finalize_stage156_fullscale_body_training.py"
RUN_ROOT="$BASE/runs/attributes/ATTR-STAGE156-FULLSCALE-BODY-TRAIN-R1"
RUN288="$RUN_ROOT/ATTR-STAGE156-BODY-CONVNEXT-288-LETTERBOX-R1"
RUN256="$RUN_ROOT/ATTR-STAGE156-BODY-CONVNEXT-256-STRETCH-R1"
STATE="$RUN_ROOT.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

[[ ! -e "$RUN_ROOT" && ! -e "$STATE" && ! -e "$STATE.sha256" ]] || {
  echo "refusing to overwrite Stage156 training evidence" >&2
  exit 66
}
assert_sha256 "$TRAINER" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST_STATE" 8d23a8a9ee5715b0613acd99f82e81f2b616d1be5837e1e8edbbf078586de228
assert_sha256 "$MANIFEST" d9f4e6f8718f389f1669b20933bb5089382bc84fba23dcd0f89a0bb5166194d5
assert_sha256 "$INIT" 6c5bdc70b4bd747345e5c96519aebad2c6eadb444458553795d1440b491c6099
assert_sha256 "$FINALIZER" 98bf5fb5ee0bc1860eeda0c46ead9321c25b2979ce83066fd4c1cfa4467f1ded
assert_sha256 "$FINALIZER_TEST" d928892ab5034a2c046cef4c501dd3d9a047198fa8c1f3d93e345cd6caf33dd0
"$PY" -m py_compile "$TRAINER" "$FINALIZER"
cd "$CODE/scripts"
"$PY" "$FINALIZER_TEST"
mkdir -p "$RUN_ROOT"

common_args=(
  --manifest "$MANIFEST"
  --labels "$LABELS"
  --architecture convnext_tiny
  --epochs 8
  --workers 8
  --learning-rate 0.0000015
  --weight-decay 0.0001
  --body-loss-weight 1.0
  --color-loss-weight 0.0
  --focal-gamma 1.5
  --color-focal-gamma 0.0
  --class-weighting inverse_sqrt
  --label-smoothing 0.01
  --gradient-clip-norm 5.0
  --freeze-backbone-epochs 0
  --patience 4
  --type-threshold 0.80
  --color-threshold 0.99
  --gate-type-precision 0.93
  --gate-type-coverage 0.45
  --seed 20260830
  --augmentation-profile hard_scene
  --selection-head body
  --body-hierarchy truck_family
  --truck-subtype-threshold 0.65
  --coarse-car-loss-weight 0.20
  --coarse-truck-loss-weight 0.0
  --night-sample-weight 5.1
  --occlusion-sample-weight 1.25
  --hard-sample-weight 0.2
  --small-sample-weight 1.25
  --color-sample-weight 0.0
  --pseudo-label-weight 1.0
  --init-checkpoint "$INIT"
  --run-kind formal
  --skip-test
)

"$PY" "$TRAINER" "${common_args[@]}" \
  --input-size 288 --resize-mode letterbox --batch-size 32 \
  --dataset-version attribute-domain-v2-stage156-fullscale-body-repair-r1 \
  --code-revision stage156-fullscale-body-288-letterbox-r1 \
  --output-dir "$RUN288"

"$PY" "$TRAINER" "${common_args[@]}" \
  --input-size 256 --resize-mode stretch --batch-size 40 \
  --dataset-version attribute-domain-v2-stage156-fullscale-body-repair-r1 \
  --code-revision stage156-fullscale-body-256-stretch-r1 \
  --output-dir "$RUN256"

"$PY" "$FINALIZER" \
  --manifest-state "$MANIFEST_STATE" --expected-manifest-state-sha256 8d23a8a9ee5715b0613acd99f82e81f2b616d1be5837e1e8edbbf078586de228 \
  --candidate-288-dir "$RUN288" --candidate-256-dir "$RUN256" \
  --trainer "$TRAINER" --expected-trainer-sha256 ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1 \
  --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --initialization-checkpoint "$INIT" --expected-initialization-sha256 6c5bdc70b4bd747345e5c96519aebad2c6eadb444458553795d1440b491c6099 \
  --output "$STATE"
