#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code/stage169-integrated-candidate-r1
ASSEMBLER=$CODE/scripts/assemble_stage169_integrated_candidate.py
ASSEMBLER_TEST=$CODE/scripts/test_assemble_stage169_integrated_candidate.py
BODY_STATE=$BASE/runs/attributes/ATTR-STAGE168-BODY-VALIDATION-R1.state.json
COLOR_STATE=$BASE/runs/attributes/ATTR-STAGE159-COMPONENT-VALIDATION-R8.state.json
LABELS=$BASE/code/config/vehicle_labels.v2.json
PRODUCTION_CONFIG=$BASE/code/stage159_component_validation_r6/production_vehicle_analytics.yaml
OUTPUT_ROOT=$BASE/runs/attributes/ATTR-STAGE169-INTEGRATED-CANDIDATE-R1
STATE=$OUTPUT_ROOT/integrated_candidate.json
OUTER_LOG=$BASE/runs/attributes/ATTR-STAGE169-INTEGRATED-CANDIDATE-R1.log

test ! -e "$OUTPUT_ROOT"
exec > >(tee -a "$OUTER_LOG") 2>&1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$ASSEMBLER" 90b6f3b1af77edb2bf2d9713a8be8ffb0b236c2afd89f8c513805566acc8042a
assert_sha256 "$ASSEMBLER_TEST" 5a4bda6da02781851507e5a5f2a167322aa2a755a5556274b354f28769ec112e
assert_sha256 "$COLOR_STATE" 87cfd43df79fd32243d06ed954d4b36aebc3724eef40f3a0ebf20d94d27a8d92
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$PRODUCTION_CONFIG" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa

cd "$CODE/scripts"
"$PY" -m unittest -v test_assemble_stage169_integrated_candidate.py
"$PY" -m py_compile "$ASSEMBLER"

while tmux has-session -t VCAS-STAGE168-BODY-VALIDATION-R1 2>/dev/null; do sleep 30; done
while pgrep -f '[e]valuate_stage166_selective_short_tracks.py .*ATTR-STAGE168' >/dev/null 2>&1; do sleep 15; done
test -s "$BODY_STATE"

set +e
"$PY" "$ASSEMBLER" \
  --body-state "$BODY_STATE" \
  --color-state "$COLOR_STATE" --expected-color-state-sha256 87cfd43df79fd32243d06ed954d4b36aebc3724eef40f3a0ebf20d94d27a8d92 \
  --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --production-config "$PRODUCTION_CONFIG" --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
  --output "$STATE"
code=$?
set -e
test "$code" = 0 -o "$code" = 2
find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
