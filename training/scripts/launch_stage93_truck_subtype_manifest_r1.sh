#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
BUILDER="$CODE/training/scripts/build_stage93_truck_subtype_manifest.py"
TESTS="$CODE/training/tests"
LABELS="$CODE/training/config/vehicle_labels.truck-subtype.v1.json"
STAGE83="$BASE/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv"
STAGE89="$BASE/datasets/attribute-domain-v2/stage89-mio-balanced-expansion-r4/attribute_manifest.stage89-mio-balanced.csv"
STAGE91="$BASE/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/attribute_manifest.stage91-all.csv"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage93-truck-subtype-r1"
MANIFEST="$OUTPUT/attribute_manifest.stage93-truck-subtype.csv"
REPORT="$OUTPUT/stage93-truck-subtype-report.json"
LOG="$BASE/runs/attributes/ATTR-STAGE93-TRUCK-SUBTYPE-MANIFEST-R1.log"
SESSION=VCAS-STAGE93-TRUCK-MANIFEST-R1

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

for required in "$PY" "$BUILDER" "$LABELS" "$STAGE83" "$STAGE89" "$STAGE91"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$BUILDER" 9dfb9cd31695066ad89b09ce337c508f87e1b1796ab56f7f4b8d6570234f539e
assert_sha256 "$CODE/training/tests/test_build_stage93_truck_subtype_manifest.py" 1594c65e571b0898fb3bbe8244e5f99cd8ff4af97a2b707972f1b76483e1b191
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$STAGE83" f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef
assert_sha256 "$STAGE89" acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7
assert_sha256 "$STAGE91" d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1

[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite: $OUTPUT" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" -m unittest discover -s "$TESTS" -p 'test_build_stage93_truck_subtype_manifest.py'
"$PY" -m py_compile "$BUILDER"

tmux new-session -d -s "$SESSION" \
  "'$PY' '$BUILDER' \
    --stage83-manifest '$STAGE83' \
    --expected-stage83-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
    --stage89-manifest '$STAGE89' \
    --expected-stage89-sha256 acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7 \
    --stage91-manifest '$STAGE91' \
    --expected-stage91-sha256 d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1 \
    --labels '$LABELS' \
    --expected-labels-sha256 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46 \
    --safety-root '$BASE/datasets' \
    --near-duplicate-hamming 4 \
    --minimum-train-per-class 10000 \
    --minimum-validation-per-class 300 \
    --output-manifest '$MANIFEST' \
    --output-report '$REPORT' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "report=$REPORT"
echo "log=$LOG"
