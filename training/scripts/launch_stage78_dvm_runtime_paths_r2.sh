#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
SCRIPT="$CODE/training/scripts/normalize_stage78_dvm_runtime_paths.py"
INPUT_ROOT="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-trainval-v1"
SOURCE_ROOT="$BASE/datasets/attribute-domain-v2/stage68-dvm-color-pool-v1"
OUTPUT_ROOT="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-runtime-v2"
INPUT_MANIFEST="$INPUT_ROOT/attribute_manifest.stage78-dvm-fine-color-trainval.csv"
INPUT_REPORT="$INPUT_ROOT/stage78-dvm-fine-color-trainval-report.json"
LABELS="$CODE/config/vehicle_labels.v2.json"
OUTPUT_MANIFEST="$OUTPUT_ROOT/attribute_manifest.stage78-dvm-fine-color-runtime.csv"
OUTPUT_REPORT="$OUTPUT_ROOT/stage78-dvm-fine-color-runtime-report.json"
LOG="$BASE/runs/attributes/ATTR-STAGE78-DVM-RUNTIME-PATHS-R2.log"
SESSION=VCAS-STAGE78-RUNTIME-R2

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

for required in "$PY" "$SCRIPT" "$INPUT_MANIFEST" "$INPUT_REPORT" "$LABELS" "$SOURCE_ROOT"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 65; }
done
assert_sha256 "$SCRIPT" 80515f9f14d10fdd42551bb4a13e72c82269ea5af415facfc622c18bd2d3ed3a
assert_sha256 "$INPUT_MANIFEST" 73ee36999a8790dcc5bac76c9d5c8e259549da304ecef8616ee1199e061c2725
assert_sha256 "$INPUT_REPORT" 8670f6c0650e04f27f38aba61baea871cead319edbba9d38092a3c2a5738da1d
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

mkdir -p "$OUTPUT_ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$SCRIPT' \
    --input-manifest '$INPUT_MANIFEST' \
    --expected-input-manifest-sha256 73ee36999a8790dcc5bac76c9d5c8e259549da304ecef8616ee1199e061c2725 \
    --input-report '$INPUT_REPORT' \
    --expected-input-report-sha256 8670f6c0650e04f27f38aba61baea871cead319edbba9d38092a3c2a5738da1d \
    --labels '$LABELS' \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --source-root '$SOURCE_ROOT' \
    --datasets-safety-root '$BASE/datasets' \
    --output-manifest '$OUTPUT_MANIFEST' \
    --output-report '$OUTPUT_REPORT' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "output_report=$OUTPUT_REPORT"
echo "log=$LOG"
