#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage156_fullscale_body_r1"
BUILDER="$CODE/scripts/build_stage156_fullscale_body_repair_manifest.py"
TEST="$CODE/scripts/test_build_stage156_fullscale_body_repair_manifest.py"
BASE_MANIFEST="$BASE/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv"
UA_MANIFEST="$BASE/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2/attribute_manifest.training-only.csv"
UA_REPORT="$BASE/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2/dataset_report.json"
NIGHT_MANIFEST="$BASE/datasets/attribute-domain-v2/stage152-lvad-night-pool-r3/attribute_manifest.stage152-lvad-night.csv"
NIGHT_REPORT="$BASE/datasets/attribute-domain-v2/stage152-lvad-night-pool-r3/stage152-lvad-night-pool-report.json"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage156-fullscale-body-repair-r1"
STATE="$OUTPUT.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

[[ ! -e "$OUTPUT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage156 evidence" >&2; exit 66; }
assert_sha256 "$BUILDER" d5dfc2c5de3cceb8fe5a7bb3bc7de21a10079a4a07ab2421601c2c51ab6c416f
assert_sha256 "$TEST" 041464d936e13e516e6cf329865d88042bb31d730f8543e8ab0a709915b33c94
assert_sha256 "$BASE_MANIFEST" f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef
assert_sha256 "$UA_MANIFEST" ab1f20ab1ef2e9da0b26b48d49efd0d4b9d324a22ec58dcdb2e102b371ebc369
assert_sha256 "$UA_REPORT" c44e4c6f0acd35ada56abed744c375b32fa6b94417f4fe7b8d693d3ad2e659ca
assert_sha256 "$NIGHT_MANIFEST" 25d8e8ff7b586cbee2622ba90eb08ab23fbcbf2acff5e496feedaff7348a6cdb
assert_sha256 "$NIGHT_REPORT" f6c8672e00db713ec3005226246150b5730a0a6b3c05c5eae8717056edf0bbc4
"$PY" -m py_compile "$BUILDER"
cd "$CODE/scripts"
"$PY" "$TEST"
"$PY" "$BUILDER" \
  --base-manifest "$BASE_MANIFEST" --expected-base-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
  --ua-manifest "$UA_MANIFEST" --expected-ua-sha256 ab1f20ab1ef2e9da0b26b48d49efd0d4b9d324a22ec58dcdb2e102b371ebc369 \
  --ua-report "$UA_REPORT" --expected-ua-report-sha256 c44e4c6f0acd35ada56abed744c375b32fa6b94417f4fe7b8d693d3ad2e659ca \
  --night-manifest "$NIGHT_MANIFEST" --expected-night-sha256 25d8e8ff7b586cbee2622ba90eb08ab23fbcbf2acff5e496feedaff7348a6cdb \
  --night-report "$NIGHT_REPORT" --expected-night-report-sha256 f6c8672e00db713ec3005226246150b5730a0a6b3c05c5eae8717056edf0bbc4 \
  --output-root "$OUTPUT" --maximum-cross-split-dhash 4 --supplement-night-fraction 0.30
