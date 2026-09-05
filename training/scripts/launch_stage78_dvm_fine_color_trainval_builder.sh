#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
BUILDER="$CODE/training/scripts/build_stage78_dvm_fine_color_trainval_manifest.py"
PARENT="$BASE/datasets/attribute-domain-v2/stage68-dvm-color-pool-v1/attribute_manifest.audit-clean-v2.csv"
OUTPUT_ROOT="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-trainval-v1"
MANIFEST="$OUTPUT_ROOT/attribute_manifest.stage78-dvm-fine-color-trainval.csv"
REPORT="$OUTPUT_ROOT/stage78-dvm-fine-color-trainval-report.json"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE78-DVM-FINE-TRAINVAL

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

for required in "$PY" "$BUILDER" "$PARENT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$BUILDER" c78b575c888c9aa957b71a4f972140813199df6a91c215afd10d12462e10b59c
assert_sha256 "$PARENT" 9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

mkdir -p "$OUTPUT_ROOT"
tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$BUILDER' \
    --parent-manifest '$PARENT' \
    --expected-parent-sha256 9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b \
    --output-manifest '$MANIFEST' \
    --output-report '$REPORT' >'$LOG' 2>&1; \
  rc=\$?; manifest_sha=''; report_sha=''; \
  if [[ -f '$MANIFEST' ]]; then manifest_sha=\$(sha256sum '$MANIFEST' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$manifest_sha\" '$(basename "$MANIFEST")' >'$MANIFEST.sha256'; fi; \
  if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$report_sha\" '$(basename "$REPORT")' >'$REPORT.sha256'; fi; \
  '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_data_only\" if int(sys.argv[2]) == 0 else \"failed_closed\",\"exit_code\":int(sys.argv[2]),\"manifest\":sys.argv[3],\"manifest_sha256\":sys.argv[4],\"report\":sys.argv[5],\"report_sha256\":sys.argv[6],\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$MANIFEST' \"\$manifest_sha\" '$REPORT' \"\$report_sha\"; \
  exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
