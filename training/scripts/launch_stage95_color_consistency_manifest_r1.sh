#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
BUILDER="$CODE/training/scripts/build_stage95_color_consistency_manifest.py"
TEST="$CODE/training/tests/test_build_stage95_color_consistency_manifest.py"
STAGE78="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-runtime-v2/attribute_manifest.stage78-dvm-fine-color-runtime.csv"
STAGE61="$BASE/datasets/attribute-domain-v2/openimages-stage61-combined-adverse-proposals-v1/attribute-proposals.csv"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage95-color-consistency-r1"
MANIFEST="$OUTPUT/attribute_manifest.stage95-color-consistency.csv"
REPORT="$OUTPUT/stage95-color-consistency-report.json"
LOG="$BASE/runs/attributes/ATTR-STAGE95-COLOR-CONSISTENCY-MANIFEST-R1.log"
SESSION=VCAS-STAGE95-COLOR-MANIFEST-R1

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

for required in "$PY" "$BUILDER" "$TEST" "$STAGE78" "$STAGE61"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$BUILDER" fd35588aaaac61cb4a30e95ccabfd6246a2466117758ffe6c8fe1c46631823d0
assert_sha256 "$TEST" f513acd1ae728d2dba6a2499ae7aa35b880f0a960841b213d2576681a02bca43
assert_sha256 "$STAGE78" dc06fe38daa5aaea1daf0c38133265e9d7f0650b5aed32f613462a1d94f1edb5
assert_sha256 "$STAGE61" db2e53b2440f4bc03a7b845a938fbf1a092e557e9650b3d2bed1a6c69f34b136

[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite: $OUTPUT" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" -m unittest discover -s "$CODE/training/tests" -p 'test_build_stage95_color_consistency_manifest.py'
"$PY" -m py_compile "$BUILDER"

tmux new-session -d -s "$SESSION" \
  "'$PY' '$BUILDER' \
    --stage78-manifest '$STAGE78' \
    --expected-stage78-sha256 dc06fe38daa5aaea1daf0c38133265e9d7f0650b5aed32f613462a1d94f1edb5 \
    --stage61-manifest '$STAGE61' \
    --expected-stage61-sha256 db2e53b2440f4bc03a7b845a938fbf1a092e557e9650b3d2bed1a6c69f34b136 \
    --datasets-root '$BASE/datasets/attribute-domain-v2' \
    --near-duplicate-hamming 4 \
    --minimum-rows 3000 \
    --minimum-night-rows 1500 \
    --output-manifest '$MANIFEST' \
    --output-report '$REPORT' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "report=$REPORT"
echo "log=$LOG"
