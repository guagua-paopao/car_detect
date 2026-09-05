#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage155_eval_r1"
BUILDER="$CODE/scripts/build_stage155_new_unused_holdout.py"
EVALUATOR="$CODE/scripts/evaluate_stage155_new_holdout_once.py"
EVAL_BASE="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
INTEGRATED_STATE="$BASE/runs/attributes/ATTR-STAGE154-INTEGRATED-COMPONENT-GATE-R2.state.json"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"

MIO_ARCHIVE="$BASE/sources/mio-tcd/MIO-TCD-Classification.tar"
VR_ARCHIVE="$BASE/sources/vehicle-rear/data.tgz"
VR_PLAN="$BASE/sources/vehicle-rear-stage149/stage149-selection-plan-r2.csv"
VR_METADATA="$BASE/sources/vehicle-rear-stage149/extracted.partial/data/dataset_3.json"
VR_README="$BASE/sources/vehicle-rear-stage149/OFFICIAL_README.md"
VR_LICENSE="$BASE/sources/vehicle-rear-stage149/OFFICIAL_LICENSE"
REF_BODY="$BASE/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv"
REF_MIO="$BASE/datasets/attribute-domain-v2/stage136-mio-road-domain-axle-r1/attribute_manifest.stage136-mio-road-domain-axle.csv"
REF_COLOR="$BASE/datasets/attribute-domain-v2/stage150-vehicle-rear-color-domain-r2/attribute_manifest.stage150-vehicle-rear-color-domain.csv"

HOLDOUT_ROOT="$BASE/datasets/attribute-domain-v2/stage155-new-unused-holdout-r1"
HOLDOUT_STATE="$HOLDOUT_ROOT.state.json"
EVAL_ROOT="$BASE/runs/attributes/ATTR-STAGE155-NEW-UNUSED-HOLDOUT-R1"
REPORT="$EVAL_ROOT/report.json"
STATE="$EVAL_ROOT.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

[[ ! -e "$HOLDOUT_ROOT" && ! -e "$HOLDOUT_STATE" && ! -e "$EVAL_ROOT" && ! -e "$STATE" ]] || {
  echo "refusing to overwrite Stage155 one-time evidence" >&2
  exit 66
}

assert_sha256 "$BUILDER" 66ba720aadbb183200dd8fcfe66418f9f1447d5da13924947c904e5b333bab0e
assert_sha256 "$EVALUATOR" 7fa86bb1f8ce18131eadc8812ad74c70856d23c98a2260df2717e91d25f1a813
assert_sha256 "$EVAL_BASE" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$INTEGRATED_STATE" 7e09040213d59f14fa72f2462c712721a7121a0c2535661fab4db2950ae84b44
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$MIO_ARCHIVE" 0cf70c660f399aef05d069d9c993e2e3f89e887c823c781d3064fee3b2e6979c
assert_sha256 "$VR_ARCHIVE" 511d92ed433d35e9861ced29b6b3b3a44f317122a42cb8a37fce654f5887858c
assert_sha256 "$VR_PLAN" 2d3f5b128a9f43b6203fd6f5e14ee903c9e100041e3341c03db7cd65f3e96867
assert_sha256 "$VR_METADATA" 1d20244fa46f0a6b48b3f7bbf95319456f82d0992f36dea834c5893e76f07353
assert_sha256 "$VR_README" f02ddc132d210f3378b30048daa8840cd62453b12a83d9b402e54bab6909cc1f
assert_sha256 "$VR_LICENSE" c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
assert_sha256 "$REF_BODY" f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef
assert_sha256 "$REF_MIO" a12be8b1cc38fa777bfd219b2a39f74b59092347c7e4b63b0d6e97509007645d
assert_sha256 "$REF_COLOR" 56a421f68cb2455b13707be7d2787fc05d456b385f36ec4af561703db752e63f
"$PY" -m py_compile "$BUILDER" "$EVALUATOR" "$EVAL_BASE"

"$PY" "$BUILDER" \
  --mio-archive "$MIO_ARCHIVE" --expected-mio-archive-sha256 0cf70c660f399aef05d069d9c993e2e3f89e887c823c781d3064fee3b2e6979c \
  --vehicle-rear-archive "$VR_ARCHIVE" --expected-vehicle-rear-archive-sha256 511d92ed433d35e9861ced29b6b3b3a44f317122a42cb8a37fce654f5887858c \
  --vehicle-rear-plan "$VR_PLAN" --expected-vehicle-rear-plan-sha256 2d3f5b128a9f43b6203fd6f5e14ee903c9e100041e3341c03db7cd65f3e96867 \
  --vehicle-rear-metadata "$VR_METADATA" --expected-vehicle-rear-metadata-sha256 1d20244fa46f0a6b48b3f7bbf95319456f82d0992f36dea834c5893e76f07353 \
  --vehicle-rear-readme "$VR_README" --expected-vehicle-rear-readme-sha256 f02ddc132d210f3378b30048daa8840cd62453b12a83d9b402e54bab6909cc1f \
  --vehicle-rear-license "$VR_LICENSE" --expected-vehicle-rear-license-sha256 c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 \
  --reference-manifest "$REF_BODY" --reference-manifest "$REF_MIO" --reference-manifest "$REF_COLOR" \
  --output-root "$HOLDOUT_ROOT"

holdout_state_sha="$(sha256sum "$HOLDOUT_STATE" | awk '{print tolower($1)}')"
mkdir -p "$EVAL_ROOT"
set +e
"$PY" "$EVALUATOR" \
  --integrated-state "$INTEGRATED_STATE" --expected-integrated-state-sha256 7e09040213d59f14fa72f2462c712721a7121a0c2535661fab4db2950ae84b44 \
  --holdout-state "$HOLDOUT_STATE" --expected-holdout-state-sha256 "$holdout_state_sha" \
  --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --baseline-body-threshold 0.978 --baseline-color-threshold 0.742 \
  --body-specialist-subtype-threshold 0.50 --datasets-safety-root "$BASE/datasets" \
  --device cuda --batch-size 64 --workers 8 --output "$REPORT" --state-output "$STATE"
code=$?
set -e
[[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
exit "$code"
